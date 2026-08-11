#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

__device__ __forceinline__ float warp_reduce_v4(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += ::__shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

static constexpr size_t BLOCK_DIM_V4 = 256;   // 8 warps

// ── v4: coalesce flash-decoding ──────────────────────────────────
// 在 v3 全体参与 cp.async 流水线基础上，增加 COALESCE 模板参数：
// 每个 grid block 处理 COALESCE 个连续 KV block（保持 KV cache TN=64 不变）。
// Grid.x = sum_i ceil(block_num_i / COALESCE)，从 block_ids 前缀推算
// (batch_id, start_block)。partial 结果按 task 粒度（非 block 粒度）写出，
// reduce kernel 也对应使用 task 粒度。
//
// 模板参数：
//   HD      — head_dim（128）
//   HEADS   — 每 block 服务的 Q head 数（grid.y = nh/HEADS）
//   TILE_K  — 每次 __syncthreads() 处理的 token 数
//   COALESCE — 每 grid task 合并的连续 KV block 数
template <typename T, size_t HD, size_t HEADS, size_t TILE_K, size_t COALESCE>
__global__ void flash_decoding_v4_kernel(
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
    constexpr size_t NUM_STAGES = 3;
    constexpr size_t NW = 8;
    extern __shared__ __align__(16) char _smem[];
    T     *s_k   = reinterpret_cast<T *>(_smem);
    T     *s_v   = s_k + NUM_STAGES * TILE_K * HD;
    float *s_q   = reinterpret_cast<float *>(s_v + NUM_STAGES * TILE_K * HD);
    float *s_acc = s_q + HEADS * HD;
    float *s_sum = s_acc + HEADS * NW * HD;
    float *s_max = s_sum + HEADS * NW;

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

    // ── 映射 task_id → (batch_id, start_block) ─────────────────
    // 遍历 batch 的 block 前缀和，按 ceil(block_num/COALESCE) 切 task
    size_t task_rem = blockIdx.x;
    size_t batch_id = 0;
    size_t prev_cum = 0;
    for (size_t i = 0; i < _batch_size; ++i) {
        size_t cur_cum = static_cast<size_t>(_block_ids[i * _max_block_num]);
        size_t bn = cur_cum - prev_cum;
        size_t tasks = (bn + COALESCE - 1) / COALESCE;
        if (task_rem < tasks) { batch_id = i; break; }
        task_rem -= tasks;
        prev_cum = cur_cum;
        batch_id = i + 1;  // 循环继续则更新；若最后 batch 未 break，此值保持
    }

    const int64_t *block_ids = _block_ids + batch_id * _max_block_num;
    const size_t   totlen    = static_cast<size_t>(_tot_len[batch_id]);
    const size_t   limit     = totlen - 1;

    // 计算本 batch 的 block_num 及当前 task 的 block 范围
    size_t batch_prev = (batch_id > 0)
        ? static_cast<size_t>(_block_ids[(batch_id - 1) * _max_block_num]) : 0;
    size_t batch_cur  = static_cast<size_t>(_block_ids[batch_id * _max_block_num]);
    size_t block_num  = batch_cur - batch_prev;
    size_t start_block = task_rem * COALESCE;
    size_t num_blocks_in_task = min(static_cast<size_t>(COALESCE), block_num - start_block);
    if (num_blocks_in_task == 0) return;

    // ── 加载 Q（warp 0..HEADS-1 各 load 一个 head）───────────────
    if (warp_id < HEADS) {
        const T *qp = _q + (_cut_idx[batch_id] * _nh + head_base + warp_id) * HD;
        float *sq = s_q + warp_id * HD;
        for (size_t col = lane_id; col < HD; col += 32)
            sq[col] = llaisys::utils::nvidia::cast<float>(qp[col]);
    }

    // ── 初始化 reg state ────────────────────────────────────────
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h)
        for (size_t i = lane_id; i < HD; i += 32)
            reg_acc[h][i >> 5] = 0.0f;

    constexpr size_t ELEMS_PER_BLK = 16 / sizeof(T);
    constexpr size_t BLOCKS_PER_ROW = HD / ELEMS_PER_BLK;
    const size_t tiles_per_block = (_token_num + TILE_K - 1) / TILE_K;
    const size_t total_tiles = num_blocks_in_task * tiles_per_block;

    // ── 预计算每个 block 的 K/V 指针 + token_offset ────────────
    const T *k_blk[COALESCE], *v_blk[COALESCE];
    size_t tok_off_blk[COALESCE];
    #pragma unroll
    for (size_t cb = 0; cb < num_blocks_in_task; ++cb) {
        int64_t bid = block_ids[start_block + cb + 1];
        size_t coff = static_cast<size_t>(bid) * _token_num * _nkvh;
        k_blk[cb] = _k_cache + (coff + kv_head) * HD;
        v_blk[cb] = _v_cache + (coff + kv_head) * HD;
        tok_off_blk[cb] = (start_block + cb) * _token_num;
    }

    // ── cp_load 辅助：目标 stage，源来自全局 tile idx ─────────
    auto cp_tile = [&](size_t stage, size_t g_tile) {
        size_t cb = g_tile / tiles_per_block;
        size_t it = g_tile % tiles_per_block;
        const T *ks = k_blk[cb], *vs = v_blk[cb];
        size_t toff = tok_off_blk[cb];
        for (size_t idx = tid; idx < TILE_K * BLOCKS_PER_ROW; idx += BLOCK_DIM_V4) {
            size_t t = idx / BLOCKS_PER_ROW, b = idx % BLOCKS_PER_ROW;
            size_t inner = it * TILE_K + t;
            if (inner < _token_num && toff + inner < totlen) {
                size_t off = b * ELEMS_PER_BLK;
                __pipeline_memcpy_async(s_k + (stage * TILE_K + t) * HD + off, ks + inner * _nkvh * HD + off, 16);
                __pipeline_memcpy_async(s_v + (stage * TILE_K + t) * HD + off, vs + inner * _nkvh * HD + off, 16);
            }
        }
    };

    // ── 预取 tile 0、1 ──────────────────────────────────────────
    cp_tile(0, 0);
    __pipeline_commit();
    if (total_tiles > 1) { cp_tile(1, 1); __pipeline_commit(); }

    // ── 主循环：pipeline 跨 block 边界连续，不 drain ─────────
    for (size_t g = 0; g < total_tiles; ++g) {
        if (g + 2 < total_tiles)
            cp_tile((g + 2) % NUM_STAGES, g + 2);
        __pipeline_commit();

        __pipeline_wait_prior((g + 2 < total_tiles) ? 1 : 0);
        __syncthreads();

        // 计算 tile g
        size_t cb = g / tiles_per_block;
        size_t tok_off = tok_off_blk[cb];
        size_t it = g % tiles_per_block;
        const size_t so = (g % NUM_STAGES) * TILE_K;
        for (size_t t = warp_id; t < TILE_K; t += NW) {
            const size_t tok = tok_off + it * TILE_K + t;
            if (tok <= limit && it * TILE_K + t < _token_num) {
                const T *kt_ = s_k + (so + t) * HD;
                const T *vt_ = s_v + (so + t) * HD;
#pragma unroll
                for (size_t h = 0; h < HEADS; ++h) {
                    float dot = 0.0f;
                    const float *qh = s_q + h * HD;
#pragma unroll
                    for (size_t c = lane_id; c < HD; c += 32)
                        dot += qh[c] * llaisys::utils::nvidia::cast<float>(kt_[c]);
                    dot = warp_reduce_v4(dot) * _scale;
                    float mn = fmaxf(reg_max[h], dot);
                    float a  = __expf(reg_max[h] - mn);
                    float b2 = __expf(dot - mn);
                    reg_max[h] = mn;
                    reg_sum[h] = a * reg_sum[h] + b2;
                    for (size_t c = lane_id; c < HD; c += 32)
                        reg_acc[h][c >> 5] = a * reg_acc[h][c >> 5]
                            + b2 * llaisys::utils::nvidia::cast<float>(vt_[c]);
                }
            }
        }

        __syncthreads();
    }

    // ── 8-warp 归约（warp 0/1 并行合并）──────────────────────────
#pragma unroll
    for (size_t h = 0; h < HEADS; ++h) {
        float *aw = s_acc + (h * NW + warp_id) * HD;
        for (size_t i = lane_id; i < HD; i += 32) aw[i] = reg_acc[h][i >> 5];
        s_sum[h * NW + warp_id] = reg_sum[h];
        s_max[h * NW + warp_id] = reg_max[h];
    }
    __syncthreads();

    // warp 0: 合并 2,3,4
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
    // warp 1: 合并 5,6,7，写回 smem[1]
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
#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            float *aw = s_acc + (h * NW + 1) * HD;
            for (size_t i = lane_id; i < HD; i += 32) aw[i] = reg_acc[h][i >> 5];
            s_sum[h * NW + 1] = reg_sum[h];
            s_max[h * NW + 1] = reg_max[h];
        }
    }
    __syncthreads();

    // warp 0: 合并 warp 1 的结果（含 1,5,6,7），写出最终 partial
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

        // 计算 task 级前缀（而非 block 级）作为写偏移
        size_t task_prefix = 0;
        size_t pc = 0;
        for (size_t i_ = 0; i_ < batch_id; ++i_) {
            size_t cc = static_cast<size_t>(_block_ids[i_ * _max_block_num]);
            size_t bn = cc - pc;
            task_prefix += (bn + COALESCE - 1) / COALESCE;
            pc = cc;
        }
        int64_t write_base = static_cast<int64_t>(task_prefix + task_rem);

#pragma unroll
        for (size_t h = 0; h < HEADS; ++h) {
            const size_t hd = head_base + h;
            int64_t off = (write_base * _nh + static_cast<int64_t>(hd));
            if (lane_id == 0) { _attn_max[off] = reg_max[h]; _attn_sum[off] = reg_sum[h]; }
            off = off * static_cast<int64_t>(HD);
            for (size_t i = lane_id; i < HD; i += 32)
                _attn_acc[off + static_cast<int64_t>(i)] = reg_acc[h][i >> 5];
        }
    }
}

// ── Reduce kernel（task 粒度，COALESCE 感知）─────────────────────
template <typename T, size_t HD, size_t COALESCE = 1>
__global__ void flash_decoding_v4_reduce(
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

    // 计算 task 级偏移与 task_num
    int64_t task_offset = 0;
    int64_t prev = 0;
    for (size_t i = 0; i < batch_id; ++i) {
        int64_t cur = _block_ids[i * _max_block_num];
        int64_t bn = cur - prev;
        task_offset += (bn + static_cast<int64_t>(COALESCE) - 1) / static_cast<int64_t>(COALESCE);
        prev = cur;
    }
    int64_t cur = _block_ids[batch_id * _max_block_num];
    int64_t block_num = cur - prev;
    int64_t task_num = (block_num + static_cast<int64_t>(COALESCE) - 1) / static_cast<int64_t>(COALESCE);

    int64_t base_off = task_offset * static_cast<int64_t>(_nh) + static_cast<int64_t>(nh);
    float *attn_acc = _attn_acc + base_off * static_cast<int64_t>(HD);
    float *attn_sum = _attn_sum + base_off;
    float *attn_max = _attn_max + base_off;

    size_t now = warp_id & 3;
    if (warp_id < 4) {
        for (size_t i = lane_id; i < HD; i += 32) reg_acc[i >> 5] = 0.0f;
    } else if (static_cast<int64_t>(now) < task_num) {
        float *sc = s_acc + now * HD;
        float *src = attn_acc + now * static_cast<int64_t>(_nh) * static_cast<int64_t>(HD);
        for (size_t i = lane_id; i < HD; i += 32) sc[i] = src[i];
    }
    __syncthreads();

    const int64_t padded = (task_num + 3) / 4 * 4;
    for (int64_t bid = static_cast<int64_t>(warp_id & 3); bid < padded; bid += 4) {
        if (warp_id >= 4 && bid + 4 < task_num) {
            float *sc = s_acc + (now ^ 4) * HD;
            float *src = attn_acc + (bid + 4) * static_cast<int64_t>(_nh) * static_cast<int64_t>(HD);
            for (size_t i = lane_id; i < HD; i += 32) sc[i] = src[i];
        }
        if (warp_id < 4 && bid < task_num) {
            float mc = attn_max[bid * static_cast<int64_t>(_nh)];
            float sc = attn_sum[bid * static_cast<int64_t>(_nh)];
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

// ── Host launch ──────────────────────────────────────────────────
template <typename T, size_t HD, size_t HEADS, size_t TILE_K, size_t COALESCE>
void launch_v4_kernel(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
                      const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
                      const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
                      size_t token_num, size_t batch_size, size_t max_block_num,
                      size_t tot_block_num, size_t tot_task_num,
                      float scale, size_t nh, size_t nkvh)
{
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    auto *d_acc  = reinterpret_cast<float *>(attn_acc);
    auto *d_sum  = reinterpret_cast<float *>(attn_sum);
    auto *d_max  = reinterpret_cast<float *>(attn_max);
    const auto *dq = reinterpret_cast<const T *>(q);
    const auto *dk = reinterpret_cast<const T *>(k_cache);
    const auto *dv2 = reinterpret_cast<const T *>(v_cache);
    const auto *db = reinterpret_cast<const int64_t *>(block_ids);
    const auto *dc = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *dt = reinterpret_cast<const int64_t *>(tot_len);

    dim3 grid0(static_cast<unsigned int>(tot_task_num), static_cast<unsigned int>(nh / HEADS));
    dim3 grid1(1, static_cast<unsigned int>(nh), static_cast<unsigned int>(batch_size));
    dim3 blockDim(BLOCK_DIM_V4);

    constexpr size_t NUM_STAGES = 3;
    constexpr size_t NW = 8;
    const size_t smem = NUM_STAGES * TILE_K * HD * sizeof(T) * 2   // s_k + s_v
        + HEADS * HD * sizeof(float)                       // s_q
        + HEADS * NW * HD * sizeof(float)                  // s_acc
        + 2 * HEADS * NW * sizeof(float);                  // s_sum + s_max

    auto *kernel = flash_decoding_v4_kernel<T, HD, HEADS, TILE_K, COALESCE>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    kernel<<<grid0, blockDim, smem>>>(d_acc, d_sum, d_max, dq, dk, dv2, db, dc, dt,
                                      token_num, batch_size, max_block_num, scale, nh, nkvh);
    flash_decoding_v4_reduce<T, HD, COALESCE><<<grid1, blockDim>>>(
        d_attn, d_acc, d_sum, d_max, db, max_block_num, nh);
    CUDA_CHECK(cudaGetLastError());
}

// ── 模板分发器（4 HEADS × 2 TILE_K × 4 COALESCE = 32 实例）────
template <typename T, size_t HD>
static void flash_decoding_v4_impl(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    size_t tot_block_num, size_t tot_task_num,
    float scale, size_t nh, size_t nkvh, int heads, int tile_k, int coalesce)
{
#define V4_LAUNCH(H, TK, CO)                                                           \
    launch_v4_kernel<T, HD, H, TK, CO>(attn_val, attn_acc, attn_sum, attn_max,        \
        q, k_cache, v_cache, block_ids, cut_idx, tot_len,                              \
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num, scale, nh, nkvh)

    // 4 HEADS × 2 TILE_K × 6 COALESCE = 48 路分发
    // CO=1(TN=64) 2(128) 3(192) 4(256) 6(384) 8(512)
    if (heads == 6 && tile_k == 8  && coalesce == 1) V4_LAUNCH(6, 8, 1);
    else if (heads == 6 && tile_k == 8  && coalesce == 2) V4_LAUNCH(6, 8, 2);
    else if (heads == 6 && tile_k == 8  && coalesce == 3) V4_LAUNCH(6, 8, 3);
    else if (heads == 6 && tile_k == 8  && coalesce == 4) V4_LAUNCH(6, 8, 4);
    else if (heads == 6 && tile_k == 8  && coalesce == 6) V4_LAUNCH(6, 8, 6);
    else if (heads == 6 && tile_k == 8  && coalesce == 8) V4_LAUNCH(6, 8, 8);
    else if (heads == 6 && tile_k == 16 && coalesce == 1) V4_LAUNCH(6, 16, 1);
    else if (heads == 6 && tile_k == 16 && coalesce == 2) V4_LAUNCH(6, 16, 2);
    else if (heads == 6 && tile_k == 16 && coalesce == 3) V4_LAUNCH(6, 16, 3);
    else if (heads == 6 && tile_k == 16 && coalesce == 4) V4_LAUNCH(6, 16, 4);
    else if (heads == 6 && tile_k == 16 && coalesce == 6) V4_LAUNCH(6, 16, 6);
    else if (heads == 6 && tile_k == 16 && coalesce == 8) V4_LAUNCH(6, 16, 8);
    else if (heads == 3 && tile_k == 8  && coalesce == 1) V4_LAUNCH(3, 8, 1);
    else if (heads == 3 && tile_k == 8  && coalesce == 2) V4_LAUNCH(3, 8, 2);
    else if (heads == 3 && tile_k == 8  && coalesce == 3) V4_LAUNCH(3, 8, 3);
    else if (heads == 3 && tile_k == 8  && coalesce == 4) V4_LAUNCH(3, 8, 4);
    else if (heads == 3 && tile_k == 8  && coalesce == 6) V4_LAUNCH(3, 8, 6);
    else if (heads == 3 && tile_k == 8  && coalesce == 8) V4_LAUNCH(3, 8, 8);
    else if (heads == 3 && tile_k == 16 && coalesce == 1) V4_LAUNCH(3, 16, 1);
    else if (heads == 3 && tile_k == 16 && coalesce == 2) V4_LAUNCH(3, 16, 2);
    else if (heads == 3 && tile_k == 16 && coalesce == 3) V4_LAUNCH(3, 16, 3);
    else if (heads == 3 && tile_k == 16 && coalesce == 4) V4_LAUNCH(3, 16, 4);
    else if (heads == 3 && tile_k == 16 && coalesce == 6) V4_LAUNCH(3, 16, 6);
    else if (heads == 3 && tile_k == 16 && coalesce == 8) V4_LAUNCH(3, 16, 8);
    else if (heads == 2 && tile_k == 8  && coalesce == 1) V4_LAUNCH(2, 8, 1);
    else if (heads == 2 && tile_k == 8  && coalesce == 2) V4_LAUNCH(2, 8, 2);
    else if (heads == 2 && tile_k == 8  && coalesce == 3) V4_LAUNCH(2, 8, 3);
    else if (heads == 2 && tile_k == 8  && coalesce == 4) V4_LAUNCH(2, 8, 4);
    else if (heads == 2 && tile_k == 8  && coalesce == 6) V4_LAUNCH(2, 8, 6);
    else if (heads == 2 && tile_k == 8  && coalesce == 8) V4_LAUNCH(2, 8, 8);
    else if (heads == 2 && tile_k == 16 && coalesce == 1) V4_LAUNCH(2, 16, 1);
    else if (heads == 2 && tile_k == 16 && coalesce == 2) V4_LAUNCH(2, 16, 2);
    else if (heads == 2 && tile_k == 16 && coalesce == 3) V4_LAUNCH(2, 16, 3);
    else if (heads == 2 && tile_k == 16 && coalesce == 4) V4_LAUNCH(2, 16, 4);
    else if (heads == 2 && tile_k == 16 && coalesce == 6) V4_LAUNCH(2, 16, 6);
    else if (heads == 2 && tile_k == 16 && coalesce == 8) V4_LAUNCH(2, 16, 8);
    else if (heads == 1 && tile_k == 8  && coalesce == 1) V4_LAUNCH(1, 8, 1);
    else if (heads == 1 && tile_k == 8  && coalesce == 2) V4_LAUNCH(1, 8, 2);
    else if (heads == 1 && tile_k == 8  && coalesce == 3) V4_LAUNCH(1, 8, 3);
    else if (heads == 1 && tile_k == 8  && coalesce == 4) V4_LAUNCH(1, 8, 4);
    else if (heads == 1 && tile_k == 8  && coalesce == 6) V4_LAUNCH(1, 8, 6);
    else if (heads == 1 && tile_k == 8  && coalesce == 8) V4_LAUNCH(1, 8, 8);
    else if (heads == 1 && tile_k == 16 && coalesce == 1) V4_LAUNCH(1, 16, 1);
    else if (heads == 1 && tile_k == 16 && coalesce == 2) V4_LAUNCH(1, 16, 2);
    else if (heads == 1 && tile_k == 16 && coalesce == 3) V4_LAUNCH(1, 16, 3);
    else if (heads == 1 && tile_k == 16 && coalesce == 4) V4_LAUNCH(1, 16, 4);
    else if (heads == 1 && tile_k == 16 && coalesce == 6) V4_LAUNCH(1, 16, 6);
    else if (heads == 1 && tile_k == 16 && coalesce == 8) V4_LAUNCH(1, 16, 8);
    else ASSERT(false, "flash_decoding_v4: unsupported (heads, tile_k, coalesce)");

#undef V4_LAUNCH

    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void flash_decoding_v4_dispatch(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    size_t tot_block_num, size_t tot_task_num,
    float scale, size_t nh, size_t dv, size_t d, size_t nkvh,
    int heads, int tile_k, int coalesce)
{
    ASSERT(d == dv, "flash_decoding_v4: d must equal dv.");

#define V4_IMPL(HD)                                                                    \
    flash_decoding_v4_impl<T, HD>(attn_val, attn_acc, attn_sum, attn_max,             \
        q, k_cache, v_cache, block_ids, cut_idx, tot_len,                              \
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num,             \
        scale, nh, nkvh, heads, tile_k, coalesce)

    switch (d) {
    case 128: V4_IMPL(128); break;
    case 64:  V4_IMPL(64);  break;
    case 32:  V4_IMPL(32);  break;
    case 16:  V4_IMPL(16);  break;
    case 8:   V4_IMPL(8);   break;
    default:  ASSERT(false, "flash_decoding_v4: unsupported head_dim.");
    }

#undef V4_IMPL
}

// ── v4 主入口（由 v3 flash_decoding() 在 coalesce>1 时路由）─────
namespace llaisys::ops::nvidia {
void flash_decoding_v4(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
                       const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
                       const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
                       const size_t token_num, const size_t batch_size, const size_t max_block_num,
                       const size_t tot_block_num, const size_t tot_task_num, llaisysDataType_t dtype,
                       const float &scale, const size_t &nh, const size_t &dv, const size_t &d,
                       const size_t &nkvh, int heads, int tile_k, int coalesce)
{
    // auto-pick HEADS/TILE_K (复用 v3 启发式，但对 coalesce>1 可用更大的 tile)
    if (heads == 0 || tile_k == 0) {
        const size_t rep = nh / nkvh;
        const size_t tokens = tot_block_num * token_num;
        if (rep == 6) {
            if (batch_size == 1 && tokens <= 1024) {
                heads = 2; tile_k = 16;
            } else if (tokens <= 512) {
                heads = 2;
            } else if (tokens <= 2048) {
                heads = 3;
            } else {
                heads = 6;
            }
        } else {
            heads = 1;
        }
        if (tile_k == 0) {
            const size_t blocks = tot_task_num * (nh / static_cast<size_t>(heads));
            tile_k = (blocks < 64) ? 16 : 8;
        }
    }

    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        flash_decoding_v4_dispatch<float>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
            scale, nh, dv, d, nkvh, heads, tile_k, coalesce);
        break;
    case LLAISYS_DTYPE_F16:
        flash_decoding_v4_dispatch<llaisys::fp16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
            scale, nh, dv, d, nkvh, heads, tile_k, coalesce);
        break;
    case LLAISYS_DTYPE_BF16:
        flash_decoding_v4_dispatch<llaisys::bf16_t>(attn_val, attn_acc, attn_sum, attn_max, q, k_cache, v_cache,
            block_ids, cut_idx, tot_len, token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
            scale, nh, dv, d, nkvh, heads, tile_k, coalesce);
        break;
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia

// ── 预计算 tot_task_num（CPU 端从 block_ids 前缀计算）───────────
// 供 C wrapper 使用：给定 block_ids CPU 副本 + batch_size + coalesce，
// 返回 sum_i ceil(block_num_i / coalesce)。
// 注意：block_ids 布局为首列前缀和（累加块数）。
static size_t compute_tot_task_num(const int64_t *h_block_ids, size_t batch_size,
                                    size_t max_block_num, size_t coalesce)
{
    size_t tot = 0;
    int64_t prev = 0;
    for (size_t i = 0; i < batch_size; ++i) {
        int64_t cur = h_block_ids[i * max_block_num];
        int64_t bn = cur - prev;
        tot += static_cast<size_t>((static_cast<uint64_t>(bn) + coalesce - 1) / coalesce);
        prev = cur;
    }
    return tot;
}

// ── C wrapper（实验用：显式 heads/tile_k/coalesce）────────────────
extern "C" __export void llaisysFlashDecodingV4(
    void *attn_val, void *attn_acc, void *attn_sum, void *attn_max,
    void *q, void *k_cache, void *v_cache,
    void *block_ids, void *cut_idx, void *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num, size_t tot_block_num,
    size_t max_seq_len, int dtype, float scale, size_t nh, size_t dv, size_t d, size_t nkvh,
    int heads, int tile_k, int coalesce)
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

    // 从 GPU block_ids 读取前缀和，在 host 端计算 tot_task_num
    // 需要一个临时 host buffer
    std::vector<int64_t> h_block_ids(batch_size * max_block_num);
    cudaMemcpy(h_block_ids.data(), block_ids,
               batch_size * max_block_num * sizeof(int64_t), cudaMemcpyDeviceToHost);
    size_t tot_task_num = compute_tot_task_num(h_block_ids.data(), batch_size, max_block_num,
                                                static_cast<size_t>(coalesce > 0 ? coalesce : 1));

    int h = heads, tk = tile_k, co = coalesce;
    if (co <= 0) co = 1;

    switch (dtype) {
    case 0: flash_decoding_v4_dispatch<float>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
        scale, nh, dv, d, nkvh, h, tk, co); break;
    case 1: flash_decoding_v4_dispatch<llaisys::fp16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
        scale, nh, dv, d, nkvh, h, tk, co); break;
    case 2: flash_decoding_v4_dispatch<llaisys::bf16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num,
        scale, nh, dv, d, nkvh, h, tk, co); break;
    default: ASSERT(false, "flash_decoding_v4: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
