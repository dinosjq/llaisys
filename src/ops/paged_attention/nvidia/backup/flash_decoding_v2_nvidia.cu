#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

// ── Warp reduce ───────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_v2(float val) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

static constexpr size_t BLOCK_DIM_V2 = 256;   // 8 warps

// ── 参数化 flash-decoding kernel ──────────────────────────────
// HEADS : 每 block 服务的 query head 数（x = nh / grid.y，须整除 nh，
//         且连续 HEADS 个 head 落在同一 kv_head 组内才能共享 K/V）。
// TILE_K: 双缓冲粒度（每 __syncthreads 处理的 token 数，每行 KV 只被一个 warp 用）。
// 动态 smem（支持 TILE_K 大时 >48KB）：
//   s_q[HEADS][HD] + s_k[2*TILE_K][HD] + s_v[2*TILE_K][HD] + s_acc[HEADS][4][HD] + s_sum/max[HEADS][4]
template <typename T, size_t HD, size_t HEADS, size_t TILE_K>
__global__ void flash_decoding_v2_kernel(
    float *__restrict__ _attn_acc,
    float *__restrict__ _attn_sum,
    float *__restrict__ _attn_max,
    const T *__restrict__ _q,
    const T *__restrict__ _k_cache,
    const T *__restrict__ _v_cache,
    const int64_t *_block_ids,
    const int64_t *_cut_idx,
    const int64_t *_tot_len,
    const size_t _token_num,
    const size_t _batch_size,
    const size_t _max_block_num,
    const float _scale,
    const size_t _nh,
    const size_t _nkvh)
{
    const size_t rep = _nh / _nkvh;          // GQA group size
    const size_t head_base = HEADS * blockIdx.y;  // 本 block 服务 HEADS 个连续 head
    const size_t kv_head = head_base / rep;  // 共享的 kv_head（HEADS 个 head 须在同一组）

    // ── 动态 smem ─────────────────────────────────────────────
    extern __shared__ float smem[];
    float *s_q   = smem;                            // HEADS * HD
    float *s_k   = s_q + HEADS * HD;                // 2*TILE_K * HD
    float *s_v   = s_k + 2 * TILE_K * HD;           // 2*TILE_K * HD
    float *s_acc = s_v + 2 * TILE_K * HD;           // HEADS * 4 * HD
    float *s_sum = s_acc + HEADS * 4 * HD;          // HEADS * 4
    float *s_max = s_sum + HEADS * 4;               // HEADS * 4

    // ── per-thread registers: HEADS 个 head 各一份 ─────────────
    float reg_acc[HEADS][(HD + 31) >> 5];   // ceil(HD/32)，HD<32 时至少 1 个
    float reg_sum[HEADS], reg_max[HEADS];
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h) {
        reg_sum[h] = 0.0f;
        reg_max[h] = -INFINITY;
    }

    const size_t tid      = threadIdx.x;
    const size_t warp_id  = tid >> 5;
    const size_t lane_id  = tid & 31;
    const bool   is_comp  = (warp_id < 4);
    const bool   is_prod  = (warp_id >= 4);

    // ── grid.x 全局块索引 → (batch_id, local table_id) ─────────
    size_t table_id = blockIdx.x;
    size_t batch_id = 0;
    for (size_t i = 0; i < _batch_size; ++i) {
        if (table_id < static_cast<size_t>(_block_ids[i * _max_block_num])) {
            batch_id = i;
            if (i > 0)
                table_id -= static_cast<size_t>(_block_ids[(i - 1) * _max_block_num]);
            break;
        }
    }

    const int64_t *block_ids = _block_ids + batch_id * _max_block_num;
    const size_t   totlen    = _tot_len[batch_id];
    const size_t   limit     = totlen - 1;

    const int64_t block_num = (batch_id > 0)
        ? (_block_ids[batch_id * _max_block_num] - _block_ids[(batch_id - 1) * _max_block_num])
        : _block_ids[0];
    if (table_id >= static_cast<size_t>(block_num)) return;

    const int64_t block_id     = block_ids[table_id + 1];
    const size_t  cache_offset = block_id * _token_num * _nkvh;
    const size_t  token_offset = table_id * _token_num;

    const T *k_cache = _k_cache + (cache_offset + kv_head) * HD;
    const T *v_cache = _v_cache + (cache_offset + kv_head) * HD;

    // ── load HEADS 个 head 的 Q（warp 0..HEADS-1 各 load 一个，
    //    不限于 consumer warp——HEADS 可能大于 4，producer 也要帮忙）──
    if (warp_id < HEADS) {
        const T *qp = _q + (_cut_idx[batch_id] * _nh + head_base + warp_id) * HD;
        float *sq = s_q + warp_id * HD;
        for (size_t col = lane_id; col < HD; col += 32)
            sq[col] = llaisys::utils::nvidia::cast<float>(qp[col]);
    }

    // ── init reg state ─────────────────────────────────────────
    if (is_comp) {
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h)
            for (size_t i = lane_id; i < HD; i += 32)
                reg_acc[h][i >> 5] = 0.0f;
    }

    // ── 预取首个 TILE_K-token tile 到 stage 0 ──────────────────
    size_t stage = 0;
    if (is_prod) {
        const size_t pw = warp_id - 4;   // 0..3
        for (size_t t = pw; t < TILE_K; t += 4) {
            if (t < _token_num && token_offset + t < totlen) {
                const T *ku = k_cache + t * _nkvh * HD;
                const T *vu = v_cache + t * _nkvh * HD;
                float *sk = s_k + t * HD, *sv = s_v + t * HD;
#pragma unroll
                for (size_t col = lane_id; col < HD; col += 32) {
                    if (col < HD) sk[col] = llaisys::utils::nvidia::cast<float>(ku[col]);
                }
#pragma unroll
                for (size_t col = lane_id; col < HD; col += 32) {
                    if (col < HD) sv[col] = llaisys::utils::nvidia::cast<float>(vu[col]);
                }
            }
        }
    }
    __syncthreads();

    // ── tile 主循环: 每 TILE_K token 一次 sync，HEADS 个 head 并行 ──
    for (size_t tile_start = 0; tile_start < _token_num; tile_start += TILE_K) {

        // consumer: 4 warps × TILE_K/4 token × HEADS head
        if (is_comp) {
            const size_t so = stage * TILE_K;
            for (size_t t = warp_id; t < TILE_K; t += 4) {
                const size_t tok = token_offset + tile_start + t;
                if (tok <= limit && (tile_start + t) < _token_num) {
                    const float *kt_ = s_k + (so + t) * HD;
                    const float *vt_ = s_v + (so + t) * HD;
#pragma unroll
                    for (size_t h = 0; h < HEADS; ++h) {
                        float dot = 0.0f;
                        const float *qh = s_q + h * HD;
#pragma unroll
                        for (size_t i = lane_id; i < HD; i += 32) dot += qh[i] * kt_[i];
                        dot = warp_reduce_v2(dot) * _scale;
                        float mn = fmaxf(reg_max[h], dot);
                        float a  = __expf(reg_max[h] - mn);
                        float b2 = __expf(dot - mn);
                        reg_max[h] = mn;
                        reg_sum[h] = a * reg_sum[h] + b2;
                        for (size_t i = lane_id; i < HD; i += 32)
                            reg_acc[h][i >> 5] = a * reg_acc[h][i >> 5] + b2 * vt_[i];
                    }
                }
            }
        }

        // producer: 预取下个 TILE_K-token tile
        const size_t nxt = stage ^ 1;
        if (is_prod && tile_start + TILE_K < _token_num) {
            const size_t pw = warp_id - 4;
            for (size_t t = pw; t < TILE_K; t += 4) {
                const size_t inner = tile_start + TILE_K + t;
                if (inner < _token_num && token_offset + inner < totlen) {
                    const T *ku = k_cache + inner * _nkvh * HD;
                    const T *vu = v_cache + inner * _nkvh * HD;
                    float *sk = s_k + (nxt * TILE_K + t) * HD;
                    float *sv = s_v + (nxt * TILE_K + t) * HD;
#pragma unroll
                    for (size_t col = lane_id; col < HD; col += 32) {
                        if (col < HD) sk[col] = llaisys::utils::nvidia::cast<float>(ku[col]);
                    }
#pragma unroll
                    for (size_t col = lane_id; col < HD; col += 32) {
                        if (col < HD) sv[col] = llaisys::utils::nvidia::cast<float>(vu[col]);
                    }
                }
            }
        }
        __syncthreads();
        stage = nxt;
    }

    // ── warp reduce + 写 partial（HEADS 个 head）────────────────
    if (warp_id > 0 && warp_id < 4) {
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            float *aw = s_acc + (h * 4 + warp_id) * HD;
            for (size_t i = lane_id; i < HD; i += 32) aw[i] = reg_acc[h][i >> 5];
            s_sum[h * 4 + warp_id] = reg_sum[h];
            s_max[h * 4 + warp_id] = reg_max[h];
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        int64_t base = (batch_id > 0) ? _block_ids[(batch_id - 1) * _max_block_num] : 0;
        base += table_id;
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            // 合并 4 个 warp
            for (size_t wi = 1; wi < 4; ++wi) {
                float sc = s_sum[h * 4 + wi], mc = s_max[h * 4 + wi];
                float mn = fmaxf(reg_max[h], mc);
                float a  = __expf(reg_max[h] - mn), b2 = __expf(mc - mn);
                reg_max[h] = mn; reg_sum[h] = a * reg_sum[h] + b2 * sc;
                const float *aw = s_acc + (h * 4 + wi) * HD;
                for (size_t j = lane_id; j < HD; j += 32)
                    reg_acc[h][j >> 5] = reg_acc[h][j >> 5] * a + aw[j] * b2;
            }
            const size_t hd = head_base + h;
            int64_t off = (base * _nh + hd);
            if (lane_id == 0) { _attn_max[off] = reg_max[h]; _attn_sum[off] = reg_sum[h]; }
            off = off * HD;
            for (size_t i = lane_id; i < HD; i += 32)
                _attn_acc[off + i] = reg_acc[h][i >> 5];
        }
    }
}

// ── Reduce kernel（布局与 v1 一致）────────────────────────────
template <typename T, size_t HD = 128>
__global__ void flash_decoding_v2_reduce(
    T *__restrict__ _attn,
    float *__restrict__ _attn_acc,
    float *__restrict__ _attn_sum,
    float *__restrict__ _attn_max,
    const int64_t *_block_ids,
    const size_t _max_block_num,
    const size_t _nh)
{
    __shared__ float s_acc[8 * HD];
    __shared__ float s_sum[4];
    __shared__ float s_max[4];

    float reg_acc[(HD + 31) >> 5];
    float reg_sum = 0.0f, reg_max = -INFINITY;
    float alpha = 1.0f, beta = 0.0f;

    const size_t tid      = threadIdx.x;
    const size_t nh       = blockIdx.y;
    const size_t batch_id = blockIdx.z;
    const size_t warp_id  = tid >> 5;
    const size_t lane_id  = tid & 31;

    int64_t offset = (batch_id > 0) ? _block_ids[(batch_id - 1) * _max_block_num] : 0;
    offset = offset * _nh + nh;
    float *attn_acc = _attn_acc + offset * HD;
    float *attn_sum = _attn_sum + offset;
    float *attn_max = _attn_max + offset;

    const int64_t block_num = (batch_id > 0)
        ? (_block_ids[batch_id * _max_block_num] - _block_ids[(batch_id - 1) * _max_block_num])
        : _block_ids[0];

    size_t now = warp_id & 3;
    if (warp_id < 4) {
        for (size_t i = lane_id; i < HD; i += 32) reg_acc[i >> 5] = 0.0f;
    } else if (now < static_cast<size_t>(block_num)) {
        float *sc = s_acc + now * HD;
        float *src = attn_acc + now * _nh * HD;
        for (size_t i = lane_id; i < HD; i += 32) sc[i] = src[i];
    }
    __syncthreads();

    const size_t padded = (block_num + 3) / 4 * 4;
    for (size_t bid = warp_id & 3; bid < padded; bid += 4) {
        if (warp_id >= 4 && bid + 4 < static_cast<size_t>(block_num)) {
            float *sc = s_acc + (now ^ 4) * HD;
            float *src = attn_acc + (bid + 4) * _nh * HD;
            for (size_t i = lane_id; i < HD; i += 32) sc[i] = src[i];
        }
        if (warp_id < 4 && bid < static_cast<size_t>(block_num)) {
            float mc = attn_max[bid * _nh], sc = attn_sum[bid * _nh];
            float mn = fmaxf(reg_max, mc);
            alpha = __expf(reg_max - mn); beta = __expf(mc - mn);
            reg_max = mn; reg_sum = reg_sum * alpha + sc * beta;
            float *sc2 = s_acc + now * HD;
            for (size_t i = lane_id; i < HD; i += 32)
                reg_acc[i >> 5] = reg_acc[i >> 5] * alpha + sc2[i] * beta;
        }
        __syncthreads(); now ^= 4;
    }

    if (warp_id > 0 && warp_id < 4) {
        float *sc = s_acc + warp_id * HD;
        for (size_t i = lane_id; i < HD; i += 32) sc[i] = reg_acc[i >> 5];
        s_sum[warp_id] = reg_sum; s_max[warp_id] = reg_max;
    }
    __syncthreads();

    if (warp_id == 0) {
        for (size_t i = 1; i <= 3; ++i) {
            float mc = s_max[i], sc = s_sum[i];
            float mn = fmaxf(reg_max, mc);
            alpha = __expf(reg_max - mn); beta = __expf(mc - mn);
            reg_max = mn; reg_sum = reg_sum * alpha + sc * beta;
            float *sc2 = s_acc + i * HD;
            for (size_t j = lane_id; j < HD; j += 32)
                reg_acc[j >> 5] = reg_acc[j >> 5] * alpha + sc2[j] * beta;
        }
        float den = (reg_sum > 0.0f) ? (1.0f / reg_sum) : 1.0f;
        size_t attn_off = (batch_id * _nh + nh) * HD;
        T *attn_cur = _attn + attn_off;
        for (size_t i = lane_id; i < HD; i += 32)
            attn_cur[i] = llaisys::utils::nvidia::cast<T>(reg_acc[i >> 5] * den);
    }
}

// ── Host launch（模板分派 HD × HEADS × TILE_K）────────────────
template <typename T, size_t HD, size_t HEADS, size_t TILE_K>
void launch_v2(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
               const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
               const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
               size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
               float scale, size_t nh, size_t nkvh)
{
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    auto *d_acc  = reinterpret_cast<float *>(attn_acc);
    auto *d_sum  = reinterpret_cast<float *>(attn_sum);
    auto *d_max  = reinterpret_cast<float *>(attn_max);
    const auto *dq = reinterpret_cast<const T *>(q);
    const auto *dk = reinterpret_cast<const T *>(k_cache);
    const auto *dv = reinterpret_cast<const T *>(v_cache);
    const auto *db = reinterpret_cast<const int64_t *>(block_ids);
    const auto *dc = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *dt = reinterpret_cast<const int64_t *>(tot_len);

    dim3 grid0(tot_block_num, nh / HEADS);
    dim3 grid1(1, nh, batch_size);
    dim3 blockDim(BLOCK_DIM_V2);

    const size_t smem = sizeof(float) * (
        HEADS * HD                      // s_q
        + 2 * TILE_K * HD               // s_k
        + 2 * TILE_K * HD               // s_v
        + HEADS * 4 * HD                // s_acc
        + 2 * HEADS * 4);               // s_sum + s_max

    auto *kernel = flash_decoding_v2_kernel<T, HD, HEADS, TILE_K>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    kernel<<<grid0, blockDim, smem>>>(d_acc, d_sum, d_max, dq, dk, dv, db, dc, dt,
                                      token_num, batch_size, max_block_num, scale, nh, nkvh);
    flash_decoding_v2_reduce<T, HD><<<grid1, blockDim>>>(d_attn, d_acc, d_sum, d_max, db, max_block_num, nh);
    CUDA_CHECK(cudaGetLastError());
}

// ── 启发式自动选参：基于 (tot_block_num, token_num, nh) ──
// 规律（RTX 4060, nh=12, nkvh=2, token_num=64 全网格 nsys 扫描）:
//   HEADS: 总 KV token = tot_block_num × token_num（tot_block_num 已含整个批次）。
//          tokens ≤ 512 → H2；≤ 2048 → H3；否则 H6。
//   TILE_K: block 总数 = tot_block_num × (nh/HEADS) 是并行度。
//          block < 64 时并行不足，T16 减 sync 开销；否则 T8 保并发
//          （T16 的 32KB smem 压低每 SM 并发）。
static void pick_params(size_t tot_block_num, size_t token_num, size_t nh, size_t nkvh,
                        int &heads, int &tile_k)
{
    const size_t rep = nh / nkvh;
    const size_t tokens = tot_block_num * token_num;  // 整个批次总 KV token
    // 自动选参规律来自模型 (nh=12, nkvh=2, rep=6) 的全网格 nsys 扫描。
    // HEADS 必须整除 rep 才能让组内 head 共享 kv_head；非 rep==6 的 config
    // 回退 HEADS=1（无共享），保证通用正确性（性能优化针对真实模型）。
    if (rep == 6) {
        if (tokens <= 512)      heads = 2;
        else if (tokens <= 2048) heads = 3;
        else                     heads = 6;
    } else {
        heads = 1;
    }
    const size_t blocks = tot_block_num * (nh / static_cast<size_t>(heads));
    // nsys 精确测量: T4 只在 HEADS=1 且 block 充分时略优（H1T4），
    // 但在多 head/block（H2 等 24 blocks）下 T4 并行不足反而最差（H2T4=9.4 vs H2T16=7.1）。
    // 自动选参不选 T4；dispatch 仍支持 T4 供显式/实验使用。
    tile_k = (blocks < 64) ? 16 : 8;
}

// ── 模板 dispatch：HD 编译期，heads/tile_k 运行时 ─────────────
// 自动选参产生 (heads, tile_k) ∈ {1,2,3,6} × {8,16} 共 8 个组合
// （tile_k 由 blocks<64 决定，只能是 8 或 16）。
// T4 分支已注释：自动选参不产生 T4（nsys 证明 H2T4 在多 head/block 下并行
// 不足反而最差，H2T4=9.2 vs H2T16=6.9 @1x256）。实验需要时取消注释即可。
template <typename T, size_t HD>
static void flash_decoding_v2_impl(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    float scale, size_t nh, size_t nkvh, int heads, int tile_k)
{
#define V2_LAUNCH(H, TK)                                                               \
    launch_v2<T, HD, H, TK>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, \
                            block_ids, cut_idx, tot_len,                                \
                            token_num, batch_size, max_block_num, tot_block_num, scale, nh, nkvh)

    if (heads == 6 && tile_k == 8)  V2_LAUNCH(6, 8);
    else if (heads == 6 && tile_k == 16) V2_LAUNCH(6, 16);
    else if (heads == 3 && tile_k == 8)  V2_LAUNCH(3, 8);
    else if (heads == 3 && tile_k == 16) V2_LAUNCH(3, 16);
    else if (heads == 2 && tile_k == 8)  V2_LAUNCH(2, 8);
    else if (heads == 2 && tile_k == 16) V2_LAUNCH(2, 16);
    else if (heads == 1 && tile_k == 8)  V2_LAUNCH(1, 8);
    else if (heads == 1 && tile_k == 16) V2_LAUNCH(1, 16);
    // T4 分支（实验用，默认注释）:
    // else if (heads == 6 && tile_k == 4)  V2_LAUNCH(6, 4);
    // else if (heads == 3 && tile_k == 4)  V2_LAUNCH(3, 4);
    // else if (heads == 2 && tile_k == 4)  V2_LAUNCH(2, 4);
    // else if (heads == 1 && tile_k == 4)  V2_LAUNCH(1, 4);
    else ASSERT(false, "flash_decoding_v2: unsupported (heads, tile_k) — auto uses {1,2,3,6}x{8,16}");

#undef V2_LAUNCH

    CUDA_CHECK(cudaGetLastError());
}

// ── 运行时 d/dv → 编译期 HD 分派 ─────────────────────────────
// kernel 模板用同一 HD 做 d 和 dv，故要求 d==dv（attention head dim 恒等）。
// 用 switch(d) 替代 if/else 链 + 宏消除重复参数。
template <typename T>
static void flash_decoding_v2_dispatch(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    float scale, size_t nh, size_t dv, size_t d, size_t nkvh, int heads, int tile_k)
{
    if (heads == 0 || tile_k == 0)
        pick_params(tot_block_num, token_num, nh, nkvh, heads, tile_k);

    ASSERT(d == dv, "flash_decoding_v2: d must equal dv (head_dim).");

#define V2_IMPL(HD)                                                                    \
    flash_decoding_v2_impl<T, HD>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, \
        block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num, \
        scale, nh, nkvh, heads, tile_k)

    switch (d) {
    case 128: V2_IMPL(128); break;
    case 64:  V2_IMPL(64);  break;
    case 32:  V2_IMPL(32);  break;
    case 16:  V2_IMPL(16);  break;
    case 8:   V2_IMPL(8);   break;
    default:  ASSERT(false, "flash_decoding_v2: unsupported head_dim (d==dv in {8,16,32,64,128}).");
    }

#undef V2_IMPL
}

// ── 主接口已由 flash_decoding_v3_nvidia.cu 接管（v3 = cp.async 全体参与）──
// 本文件保留 v2 kernel + C wrapper (llaisysFlashDecodingV2) 作备份/实验。
// v2 的 nvidia::flash_decoding 定义已移除，避免与 v3 重复定义。

// ── C wrapper（实验/显式参数用；传 0 表示自动）─────────────────
extern "C" __export void llaisysFlashDecodingV2(
    void *attn_val, void *attn_acc, void *attn_sum, void *attn_max,
    void *q, void *k_cache, void *v_cache,
    void *block_ids, void *cut_idx, void *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    size_t max_seq_len, int dtype, float scale, size_t nh, size_t dv, size_t d, size_t nkvh,
    int heads, int tile_k)
{
    auto *ba  = reinterpret_cast<std::byte *>(attn_val);
    auto *bac = reinterpret_cast<std::byte *>(attn_acc);
    auto *bas = reinterpret_cast<std::byte *>(attn_sum);
    auto *bam = reinterpret_cast<std::byte *>(attn_max);
    auto *bq  = reinterpret_cast<const std::byte *>(q);
    auto *bkc = reinterpret_cast<const std::byte *>(k_cache);
    auto *bvc = reinterpret_cast<const std::byte *>(v_cache);
    auto *bb  = reinterpret_cast<const std::byte *>(block_ids);
    auto *bc  = reinterpret_cast<const std::byte *>(cut_idx);
    auto *bt  = reinterpret_cast<const std::byte *>(tot_len);

    switch (dtype) {
    case 0: flash_decoding_v2_dispatch<float>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    case 1: flash_decoding_v2_dispatch<llaisys::fp16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    case 2: flash_decoding_v2_dispatch<llaisys::bf16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    default: ASSERT(false, "flash_decoding_v2: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
