#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

__device__ __forceinline__ float warp_reduce_v3(float val) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

static constexpr size_t BLOCK_DIM_V3 = 256;   // 8 warps

// ── v3: cp.async 全体参与流水线 flash-decoding ────────────────
// 与 v2 相同结构（grid=(tot_block, nh/HEADS)、HEADS 个 head 共享 kv_head），
// 但**取消 producer/consumer 分离**——所有 8 个 warp 都发 cp.async 加载
// 自己负责的 K/V 块、等自己的 load、算自己的 token 部分。sync 处各线程
// 等待均匀（无 producer 等、consumer 空等的不平衡）。
//   * s_k/s_v 存原始 T 类型（cp.async 直拷 16B 块），计算时 cast
//   * 每 warp 处理 TILE_K/8 个 token × HEADS head，8-warp 归约
template <typename T, size_t HD, size_t HEADS, size_t TILE_K>
__global__ void flash_decoding_v3_kernel(
    float *__restrict__ _attn_acc, float *__restrict__ _attn_sum, float *__restrict__ _attn_max,
    const T *__restrict__ _q, const T *__restrict__ _k_cache, const T *__restrict__ _v_cache,
    const int64_t *_block_ids, const int64_t *_cut_idx, const int64_t *_tot_len,
    const size_t _token_num, const size_t _batch_size, const size_t _max_block_num,
    const float _scale, const size_t _nh, const size_t _nkvh)
{
    const size_t rep = _nh / _nkvh;
    const size_t head_base = HEADS * blockIdx.y;
    const size_t kv_head = head_base / rep;

    // ── 动态 smem（16B 对齐）────────────────────────────────────
    constexpr size_t NUM_STAGES = 3;          // prefetch i+2 写 (i+2)%3
    constexpr size_t NW = 8;                  // 8 warps 全体参与
    extern __shared__ __align__(16) char _smem[];
    T     *s_k   = reinterpret_cast<T *>(_smem);                    // NUM_STAGES*TILE_K*HD
    T     *s_v   = s_k + NUM_STAGES * TILE_K * HD;                  // NUM_STAGES*TILE_K*HD
    float *s_q   = reinterpret_cast<float *>(s_v + NUM_STAGES * TILE_K * HD);   // HEADS*HD
    float *s_acc = s_q + HEADS * HD;                                // HEADS*NW*HD
    float *s_sum = s_acc + HEADS * NW * HD;                         // HEADS*NW
    float *s_max = s_sum + HEADS * NW;                              // HEADS*NW

    float reg_acc[HEADS][(HD + 31) >> 5];
    float reg_sum[HEADS], reg_max[HEADS];
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h) {
        reg_sum[h] = 0.0f;
        reg_max[h] = -INFINITY;
    }

    const size_t tid      = threadIdx.x;
    const size_t warp_id  = tid >> 5;
    const size_t lane_id  = tid & 31;

    // ── grid.x → (batch_id, local table_id) ────────────────────
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

    // ── load Q（同步，量小；warp 0..HEADS-1 各 load 一个）──────
    if (warp_id < HEADS) {
        const T *qp = _q + (_cut_idx[batch_id] * _nh + head_base + warp_id) * HD;
        float *sq = s_q + warp_id * HD;
        for (size_t col = lane_id; col < HD; col += 32)
            sq[col] = llaisys::utils::nvidia::cast<float>(qp[col]);
    }

    // ── init reg state（所有 8 warp）────────────────────────────
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h)
        for (size_t i = lane_id; i < HD; i += 32)
            reg_acc[h][i >> 5] = 0.0f;

    // ── cp.async: 所有 256 线程分摊发 K/V（16B 块）─────────────
    // 每 16B = 16/sizeof(T) 个元素。BLOCKS_PER_ROW = HD/ELEMS_PER_BLK。
    constexpr size_t ELEMS_PER_BLK = 16 / sizeof(T);
    constexpr size_t BLOCKS_PER_ROW = HD / ELEMS_PER_BLK;

    auto cp_load = [&](size_t stage, size_t tile_start) {
        const size_t total_blk = TILE_K * BLOCKS_PER_ROW;    // 本 stage K 总块数
        for (size_t idx = tid; idx < total_blk; idx += BLOCK_DIM_V3) {
            const size_t t = idx / BLOCKS_PER_ROW;           // 行内 token
            const size_t b = idx % BLOCKS_PER_ROW;           // 行内 16B 块
            const size_t inner = tile_start + t;
            if (inner < _token_num && token_offset + inner < totlen) {
                const size_t off = b * ELEMS_PER_BLK;
                const T *ksrc = k_cache + inner * _nkvh * HD;
                const T *vsrc = v_cache + inner * _nkvh * HD;
                T *kdst = s_k + (stage * TILE_K + t) * HD;
                T *vdst = s_v + (stage * TILE_K + t) * HD;
                __pipeline_memcpy_async(kdst + off, ksrc + off, 16);
                __pipeline_memcpy_async(vdst + off, vsrc + off, 16);
            }
        }
    };

    // ── 预取 stage 0、1（lookahead 2 = NUM_STAGES-1）───────────
    const size_t num_tiles = (_token_num + TILE_K - 1) / TILE_K;
    cp_load(0, 0);
    __pipeline_commit();                 // 所有线程 commit 保持组计数一致
    if (num_tiles > 1) {
        cp_load(1, TILE_K);
        __pipeline_commit();
    }

    // ── 流水主循环（3-stage 缓冲，prefetch stage i+2 写 (i+2)%3）─
    for (size_t i = 0; i < num_tiles; ++i) {
        // 全体发 stage i+2 的 cp.async
        if (i + 2 < num_tiles) {
            cp_load((i + 2) % NUM_STAGES, (i + 2) * TILE_K);
        }
        __pipeline_commit();

        // 等 stage i 完成（最后 tile 等全部）
        __pipeline_wait_prior((i + 2 < num_tiles) ? 1 : 0);
        __syncthreads();

        // 计算 stage i：每 warp 处理 TILE_K/8 个 token（stride 8）× HEADS head
        const size_t so = (i % NUM_STAGES) * TILE_K;
        for (size_t t = warp_id; t < TILE_K; t += NW) {
            const size_t tok = token_offset + i * TILE_K + t;
            if (tok <= limit && i * TILE_K + t < _token_num) {
                const T *kt_ = s_k + (so + t) * HD;
                const T *vt_ = s_v + (so + t) * HD;
#pragma unroll
                for (size_t h = 0; h < HEADS; ++h) {
                    float dot = 0.0f;
                    const float *qh = s_q + h * HD;
#pragma unroll
                    for (size_t c = lane_id; c < HD; c += 32)
                        dot += qh[c] * llaisys::utils::nvidia::cast<float>(kt_[c]);
                    dot = warp_reduce_v3(dot) * _scale;
                    float mn = fmaxf(reg_max[h], dot);
                    float a  = __expf(reg_max[h] - mn);
                    float b2 = __expf(dot - mn);
                    reg_max[h] = mn;
                    reg_sum[h] = a * reg_sum[h] + b2;
                    for (size_t c = lane_id; c < HD; c += 32)
                        reg_acc[h][c >> 5] = a * reg_acc[h][c >> 5] + b2 * llaisys::utils::nvidia::cast<float>(vt_[c]);
                }
            }
        }

        __syncthreads();   // stage i 读完后，下一轮才能覆盖它的缓冲
    }

    // ── 8-warp 归约：只 warp 0、1 参与合并，其他只写 smem ──────
    // warp0 并 2-4，warp1 并 5-7（并行），最后 warp0 并 warp1。
    // 关键路径 7 次串行 → 3 次并行 + 1 = 4 次，只 2 次 sync。
    // 所有 8 warp 写 partial 到 smem
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h) {
        float *aw = s_acc + (h * NW + warp_id) * HD;
        for (size_t i = lane_id; i < HD; i += 32) aw[i] = reg_acc[h][i >> 5];
        s_sum[h * NW + warp_id] = reg_sum[h];
        s_max[h * NW + warp_id] = reg_max[h];
    }
    __syncthreads();

    // warp 0: 并 2,3,4（reg 已含 warp0 的 partial）
    if (warp_id == 0) {
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h)
            for (size_t wi = 2; wi <= 4; ++wi) {
                float sc = s_sum[h * NW + wi], mc = s_max[h * NW + wi];
                float mn = fmaxf(reg_max[h], mc);
                float a  = __expf(reg_max[h] - mn), b2 = __expf(mc - mn);
                reg_max[h] = mn; reg_sum[h] = a * reg_sum[h] + b2 * sc;
                const float *aw = s_acc + (h * NW + wi) * HD;
                for (size_t j = lane_id; j < HD; j += 32)
                    reg_acc[h][j >> 5] = reg_acc[h][j >> 5] * a + aw[j] * b2;
            }
    }
    // warp 1: 并 5,6,7，写回 smem[1]
    else if (warp_id == 1) {
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h)
            for (size_t wi = 5; wi <= 7; ++wi) {
                float sc = s_sum[h * NW + wi], mc = s_max[h * NW + wi];
                float mn = fmaxf(reg_max[h], mc);
                float a  = __expf(reg_max[h] - mn), b2 = __expf(mc - mn);
                reg_max[h] = mn; reg_sum[h] = a * reg_sum[h] + b2 * sc;
                const float *aw = s_acc + (h * NW + wi) * HD;
                for (size_t j = lane_id; j < HD; j += 32)
                    reg_acc[h][j >> 5] = reg_acc[h][j >> 5] * a + aw[j] * b2;
            }
        // 写回 smem[1]
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            float *aw = s_acc + (h * NW + 1) * HD;
            for (size_t i = lane_id; i < HD; i += 32) aw[i] = reg_acc[h][i >> 5];
            s_sum[h * NW + 1] = reg_sum[h];
            s_max[h * NW + 1] = reg_max[h];
        }
    }
    __syncthreads();

    // warp 0: 并 warp 1 的结果（含 1,5,6,7），现在含全部 0..7
    if (warp_id == 0) {
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            float sc = s_sum[h * NW + 1], mc = s_max[h * NW + 1];
            float mn = fmaxf(reg_max[h], mc);
            float a  = __expf(reg_max[h] - mn), b2 = __expf(mc - mn);
            reg_max[h] = mn; reg_sum[h] = a * reg_sum[h] + b2 * sc;
            const float *aw = s_acc + (h * NW + 1) * HD;
            for (size_t j = lane_id; j < HD; j += 32)
                reg_acc[h][j >> 5] = reg_acc[h][j >> 5] * a + aw[j] * b2;
        }
        // 写输出
        int64_t base = (batch_id > 0) ? _block_ids[(batch_id - 1) * _max_block_num] : 0;
        base += table_id;
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            const size_t hd = head_base + h;
            int64_t off = (base * _nh + hd);
            if (lane_id == 0) { _attn_max[off] = reg_max[h]; _attn_sum[off] = reg_sum[h]; }
            off = off * HD;
            for (size_t i = lane_id; i < HD; i += 32)
                _attn_acc[off + i] = reg_acc[h][i >> 5];
        }
    }
}

// ── Reduce kernel（与 v2 相同）────────────────────────────────
template <typename T, size_t HD = 128>
__global__ void flash_decoding_v3_reduce(
    T *__restrict__ _attn, float *__restrict__ _attn_acc, float *__restrict__ _attn_sum,
    float *__restrict__ _attn_max, const int64_t *_block_ids,
    const size_t _max_block_num, const size_t _nh)
{
    __shared__ float s_acc[8 * HD];
    __shared__ float s_sum[4];
    __shared__ float s_max[4];

    float reg_acc[(HD + 31) >> 5];
    float reg_sum = 0.0f, reg_max = -INFINITY;
    float alpha = 1.0f, beta = 0.0f;

    const size_t tid = threadIdx.x, nh = blockIdx.y, batch_id = blockIdx.z;
    const size_t warp_id = tid >> 5, lane_id = tid & 31;

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

// ── Host launch ────────────────────────────────────────────────
template <typename T, size_t HD, size_t HEADS, size_t TILE_K>
void launch_v3(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
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
    dim3 blockDim(BLOCK_DIM_V3);

    constexpr size_t NUM_STAGES = 3;
    constexpr size_t NW = 8;
    const size_t smem = NUM_STAGES * TILE_K * HD * sizeof(T) * 2   // s_k + s_v (T 类型)
        + HEADS * HD * sizeof(float)                       // s_q
        + HEADS * NW * HD * sizeof(float)                  // s_acc (8 warp)
        + 2 * HEADS * NW * sizeof(float);                  // s_sum + s_max

    auto *kernel = flash_decoding_v3_kernel<T, HD, HEADS, TILE_K>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    kernel<<<grid0, blockDim, smem>>>(d_acc, d_sum, d_max, dq, dk, dv, db, dc, dt,
                                      token_num, batch_size, max_block_num, scale, nh, nkvh);
    flash_decoding_v3_reduce<T, HD><<<grid1, blockDim>>>(d_attn, d_acc, d_sum, d_max, db, max_block_num, nh);
    CUDA_CHECK(cudaGetLastError());
}

// ── 运行时选参 → 编译期分派 ───────────────────────────────────
static void pick_params_v3(size_t tot_block_num, size_t token_num, size_t nh, size_t nkvh,
                           int &heads, int &tile_k)
{
    const size_t rep = nh / nkvh;
    const size_t tokens = tot_block_num * token_num;
    if (rep == 6) {
        if (tokens <= 512)      heads = 2;
        else if (tokens <= 2048) heads = 3;
        else                     heads = 6;
    } else {
        heads = 1;
    }
    const size_t blocks = tot_block_num * (nh / static_cast<size_t>(heads));
    tile_k = (blocks < 64) ? 16 : 8;
}

template <typename T, size_t HD>
static void flash_decoding_v3_impl(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    float scale, size_t nh, size_t nkvh, int heads, int tile_k)
{
#define V3_LAUNCH(H, TK)                                                               \
    launch_v3<T, HD, H, TK>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, \
                            block_ids, cut_idx, tot_len,                                \
                            token_num, batch_size, max_block_num, tot_block_num, scale, nh, nkvh)

    if (heads == 6 && tile_k == 8)  V3_LAUNCH(6, 8);
    else if (heads == 6 && tile_k == 16) V3_LAUNCH(6, 16);
    else if (heads == 3 && tile_k == 8)  V3_LAUNCH(3, 8);
    else if (heads == 3 && tile_k == 16) V3_LAUNCH(3, 16);
    else if (heads == 2 && tile_k == 8)  V3_LAUNCH(2, 8);
    else if (heads == 2 && tile_k == 16) V3_LAUNCH(2, 16);
    else if (heads == 1 && tile_k == 8)  V3_LAUNCH(1, 8);
    else if (heads == 1 && tile_k == 16) V3_LAUNCH(1, 16);
    else ASSERT(false, "flash_decoding_v3: unsupported (heads, tile_k)");

#undef V3_LAUNCH

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void flash_decoding_v3_dispatch(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    float scale, size_t nh, size_t dv, size_t d, size_t nkvh, int heads, int tile_k)
{
    if (heads == 0 || tile_k == 0)
        pick_params_v3(tot_block_num, token_num, nh, nkvh, heads, tile_k);
    ASSERT(d == dv, "flash_decoding_v3: d must equal dv.");

#define V3_IMPL(HD)                                                                    \
    flash_decoding_v3_impl<T, HD>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache, \
        block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num, \
        scale, nh, nkvh, heads, tile_k)

    switch (d) {
    case 128: V3_IMPL(128); break;
    case 64:  V3_IMPL(64);  break;
    case 32:  V3_IMPL(32);  break;
    case 16:  V3_IMPL(16);  break;
    case 8:   V3_IMPL(8);   break;
    default:  ASSERT(false, "flash_decoding_v3: unsupported head_dim.");
    }

#undef V3_IMPL
}

// ── 主接口（op.cpp 通过 .cuh 声明调用；自动选参）──────────────
namespace llaisys::ops::nvidia {
void flash_decoding(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
                    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
                    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
                    const size_t token_num, const size_t batch_size, const size_t max_block_num,
                    const size_t tot_block_num, const size_t max_seq_len, llaisysDataType_t dtype,
                    const float &scale, const size_t &nh, const size_t &dv, const size_t &d,
                    const size_t &nkvh)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        flash_decoding_v3_dispatch<float>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num,
            scale, nh, dv, d, nkvh, 0, 0);   // 0,0 → 自动选参
        break;
    case LLAISYS_DTYPE_F16:
        flash_decoding_v3_dispatch<llaisys::fp16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num,
            scale, nh, dv, d, nkvh, 0, 0);
        break;
    case LLAISYS_DTYPE_BF16:
        flash_decoding_v3_dispatch<llaisys::bf16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num,
            scale, nh, dv, d, nkvh, 0, 0);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia

// ── C wrapper（实验/显式参数；传 0 自动）──────────────────────
extern "C" __export void llaisysFlashDecodingV3(
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
    case 0: flash_decoding_v3_dispatch<float>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    case 1: flash_decoding_v3_dispatch<llaisys::fp16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    case 2: flash_decoding_v3_dispatch<llaisys::bf16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, scale, nh, dv, d, nkvh, heads, tile_k); break;
    default: ASSERT(false, "flash_decoding_v3: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
