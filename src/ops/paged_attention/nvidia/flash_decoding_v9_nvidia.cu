#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

// ── v9: v6 架构 (block-per-KV-block + reduce) + v8 低寄存器布局 ─
// 融合:
//   * v6: grid=(tot_task, nh/HEADS), COALESCE, block 指针预计算, cp.async
//          pipeline, reduce kernel (grid 够大)
//   * v8: blockDim=(16, HEADS), 每 thread 8 元素 q/acc 常驻寄存器,
//          shuffle width=16 归约 (低寄存器 ~72, 高 occupancy)
// 目标: v6 的 grid 并行度 + v8 的 occupancy, 克服 v6 的 159 regs → 25% 瓶颈。

__device__ __forceinline__ float subwarp_reduce_16(float val) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        val += ::__shfl_xor_sync(0xFFFFFFFF, val, offset, 16);
    return val;
}

static constexpr size_t ELEMS_PER_THREAD_V9 = 8;  // HD=128 / 16 threads

template <typename T, size_t HD, size_t HEADS, size_t TILE_K, size_t COALESCE,
          size_t NUM_STAGES = 2>
__global__ void flash_decoding_v9_kernel(
    float *__restrict__ _attn_acc, float *__restrict__ _attn_sum, float *__restrict__ _attn_max,
    const T *__restrict__ _q, const T *__restrict__ _k_cache, const T *__restrict__ _v_cache,
    const int64_t *_block_ids, const int64_t *_cut_idx, const int64_t *_tot_len,
    const size_t _token_num, const size_t _batch_size, const size_t _max_block_num,
    const float _scale, const size_t _nh, const size_t _nkvh)
{
    const size_t rep = _nh / _nkvh;
    const size_t head_base = HEADS * blockIdx.y;
    const size_t kv_head = head_base / rep;

    // 线程角色
    const size_t xseg = threadIdx.x;   // 0..15, head_dim 段
    const size_t head = threadIdx.y;   // 0..HEADS-1
    const size_t tid  = threadIdx.y * 16 + threadIdx.x;

    // ── Q 常驻寄存器 ────────────────────────────────────────────
    float q_reg[ELEMS_PER_THREAD_V9];
    float acc[ELEMS_PER_THREAD_V9];
    float reg_sum = 0.0f, reg_max = -INFINITY;

    // ── 映射 task_id → (batch_id, start_block) ──────────────────
    size_t task_rem = blockIdx.x;
    size_t batch_id = 0, prev_cum = 0;
    for (size_t i = 0; i < _batch_size; ++i) {
        size_t cur_cum = static_cast<size_t>(_block_ids[i * _max_block_num]);
        size_t bn = cur_cum - prev_cum;
        size_t tasks = (bn + COALESCE - 1) / COALESCE;
        if (task_rem < tasks) { batch_id = i; break; }
        task_rem -= tasks; prev_cum = cur_cum; batch_id = i + 1;
    }
    const int64_t *block_ids = _block_ids + batch_id * _max_block_num;
    const size_t   totlen = static_cast<size_t>(_tot_len[batch_id]);
    const size_t   limit = totlen - 1;

    size_t batch_prev = (batch_id > 0)
        ? static_cast<size_t>(_block_ids[(batch_id - 1) * _max_block_num]) : 0;
    size_t batch_cur  = static_cast<size_t>(_block_ids[batch_id * _max_block_num]);
    size_t block_num = batch_cur - batch_prev;
    size_t start_block = task_rem * COALESCE;
    size_t num_blocks_in_task = min(static_cast<size_t>(COALESCE), block_num - start_block);
    if (num_blocks_in_task == 0) return;

    // Q load (task 对应 seq 的 head)
    const T *qp = _q + (_cut_idx[batch_id] * _nh + head_base + head) * HD + xseg * ELEMS_PER_THREAD_V9;
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD_V9; ++i)
        q_reg[i] = llaisys::utils::nvidia::cast<float>(qp[i]);
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD_V9; ++i)
        acc[i] = 0.0f;

    // ── smem: 仅 s_k/s_v (T) ────────────────────────────────────
    extern __shared__ __align__(16) char _smem[];
    T *s_k = reinterpret_cast<T *>(_smem);
    T *s_v = s_k + NUM_STAGES * TILE_K * HD;

    constexpr size_t ELEMS_PER_BLK = 16 / sizeof(T);
    constexpr size_t BLOCKS_PER_ROW = HD / ELEMS_PER_BLK;
    const size_t tiles_per_block = (_token_num + TILE_K - 1) / TILE_K;
    const size_t total_tiles = num_blocks_in_task * tiles_per_block;
    const size_t BLOCK_DIM = 16 * HEADS;

    // ── 预计算 block 指针数组 ───────────────────────────────────
    const T *k_blk_arr[COALESCE], *v_blk_arr[COALESCE];
    size_t tok_off_arr[COALESCE];
#pragma unroll
    for (size_t cb = 0; cb < num_blocks_in_task; ++cb) {
        int64_t bid = block_ids[start_block + cb + 1];
        size_t coff = static_cast<size_t>(bid) * _token_num * _nkvh;
        k_blk_arr[cb] = _k_cache + (coff + kv_head) * HD;
        v_blk_arr[cb] = _v_cache + (coff + kv_head) * HD;
        tok_off_arr[cb] = (start_block + cb) * _token_num;
    }

    auto cp_tile = [&](size_t stage, size_t g_tile) {
        size_t cb = g_tile / tiles_per_block;
        size_t it = g_tile % tiles_per_block;
        const T *ks = k_blk_arr[cb], *vs = v_blk_arr[cb];
        size_t toff = tok_off_arr[cb];
        for (size_t idx = tid; idx < TILE_K * BLOCKS_PER_ROW; idx += BLOCK_DIM) {
            size_t t = idx / BLOCKS_PER_ROW, b = idx % BLOCKS_PER_ROW;
            size_t inner = it * TILE_K + t;
            if (inner < _token_num && toff + inner < totlen) {
                size_t off = b * ELEMS_PER_BLK;
                __pipeline_memcpy_async(s_k + (stage * TILE_K + t) * HD + off,
                                        ks + inner * _nkvh * HD + off, 16);
                __pipeline_memcpy_async(s_v + (stage * TILE_K + t) * HD + off,
                                        vs + inner * _nkvh * HD + off, 16);
            }
        }
    };

    // ── 预取 + flatten tile 循环 ────────────────────────────────
    cp_tile(0, 0); __pipeline_commit();

    for (size_t g = 0; g < total_tiles; ++g) {
        if (g + 1 < total_tiles)
            cp_tile((g + 1) % NUM_STAGES, g + 1);
        __pipeline_commit();
        __pipeline_wait_prior((g + 1 < total_tiles) ? 1 : 0);
        __syncthreads();

        size_t cb = g / tiles_per_block;
        size_t tok_off = tok_off_arr[cb];
        size_t it = g % tiles_per_block;
        const size_t so = (g % NUM_STAGES) * TILE_K;

        // 每行 16 threads 处理 TILE_K 个 token (串行), HEADS 行并行
        for (size_t t = 0; t < TILE_K; ++t) {
            const size_t tok = tok_off + it * TILE_K + t;
            if (tok > limit || it * TILE_K + t >= _token_num) continue;
            const T *kt = s_k + (so + t) * HD + xseg * ELEMS_PER_THREAD_V9;
            const T *vt = s_v + (so + t) * HD + xseg * ELEMS_PER_THREAD_V9;
            float dot = 0.0f;
#pragma unroll
            for (size_t i = 0; i < ELEMS_PER_THREAD_V9; ++i)
                dot += q_reg[i] * llaisys::utils::nvidia::cast<float>(kt[i]);
            dot = subwarp_reduce_16(dot) * _scale;
            float mn = fmaxf(reg_max, dot);
            float a  = __expf(reg_max - mn);
            float b2 = __expf(dot - mn);
            reg_max = mn;
            reg_sum = a * reg_sum + b2;
#pragma unroll
            for (size_t i = 0; i < ELEMS_PER_THREAD_V9; ++i)
                acc[i] = a * acc[i] + b2 * llaisys::utils::nvidia::cast<float>(vt[i]);
        }
        __syncthreads();  // stage 读完
    }

    // ── 写 partial: 每 head 16 threads 各写 8 元素段 ────────────
    size_t task_prefix = 0, pc = 0;
    for (size_t i_ = 0; i_ < batch_id; ++i_) {
        size_t cc = static_cast<size_t>(_block_ids[i_ * _max_block_num]);
        size_t bn = cc - pc;
        task_prefix += (bn + COALESCE - 1) / COALESCE;
        pc = cc;
    }
    int64_t base = static_cast<int64_t>(task_prefix + task_rem);
    const size_t hd = head_base + head;
    int64_t off = (base * _nh + static_cast<int64_t>(hd));
    if (xseg == 0) { _attn_max[off] = reg_max; _attn_sum[off] = reg_sum; }
    off = off * static_cast<int64_t>(HD) + static_cast<int64_t>(xseg) * ELEMS_PER_THREAD_V9;
    for (size_t i = 0; i < ELEMS_PER_THREAD_V9; ++i)
        _attn_acc[off + static_cast<int64_t>(i)] = acc[i];
}

// ── Reduce kernel (task 粒度, 与 v6 相同) ────────────────────────
template <typename T, size_t HD, size_t COALESCE = 1>
__global__ void flash_decoding_v9_reduce(
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

    int64_t task_offset = 0, prev = 0;
    for (size_t i = 0; i < batch_id; ++i) {
        int64_t cur = _block_ids[i * _max_block_num];
        task_offset += (cur - prev + static_cast<int64_t>(COALESCE) - 1) / static_cast<int64_t>(COALESCE);
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

// ── Host launch ────────────────────────────────────────────────────
template <typename T, size_t HD, size_t HEADS, size_t TILE_K, size_t COALESCE,
          size_t NUM_STAGES = 2>
void launch_v9(std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
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
    const auto *dv = reinterpret_cast<const T *>(v_cache);
    const auto *db = reinterpret_cast<const int64_t *>(block_ids);
    const auto *dc = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *dt = reinterpret_cast<const int64_t *>(tot_len);

    dim3 grid0(static_cast<unsigned int>(tot_task_num), static_cast<unsigned int>(nh / HEADS));
    dim3 grid1(1, static_cast<unsigned int>(nh), static_cast<unsigned int>(batch_size));
    dim3 blockDim(16, HEADS);
    dim3 blockDimRed(256);

    const size_t smem = NUM_STAGES * TILE_K * HD * sizeof(T) * 2;  // s_k + s_v

    auto *kernel = flash_decoding_v9_kernel<T, HD, HEADS, TILE_K, COALESCE, NUM_STAGES>;
    static bool attr_set = false;
    if (!attr_set) {
        cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        attr_set = true;
    }
    kernel<<<grid0, blockDim, smem>>>(d_acc, d_sum, d_max, dq, dk, dv, db, dc, dt,
                                      token_num, batch_size, max_block_num, scale, nh, nkvh);
    flash_decoding_v9_reduce<T, HD, COALESCE><<<grid1, blockDimRed>>>(
        d_attn, d_acc, d_sum, d_max, db, max_block_num, nh);
    CUDA_CHECK(cudaGetLastError());
}

// ── 模板分发: HEADS × TILE_K × COALESCE ──────────────────────────
template <typename T, size_t HD>
static void flash_decoding_v9_impl(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    size_t tot_block_num, size_t tot_task_num,
    float scale, size_t nh, size_t nkvh, int heads, int tile_k, int coalesce)
{
#define V9_LAUNCH(H, TK, CO)                                                          \
    launch_v9<T, HD, H, TK, CO>(attn_val, attn_acc, attn_sum, attn_max,              \
        q, k_cache, v_cache, block_ids, cut_idx, tot_len,                             \
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num, scale, nh, nkvh)

    if (heads == 6 && tile_k == 8  && coalesce == 1) V9_LAUNCH(6, 8, 1);
    else if (heads == 6 && tile_k == 8  && coalesce == 2) V9_LAUNCH(6, 8, 2);
    else if (heads == 6 && tile_k == 8  && coalesce == 4) V9_LAUNCH(6, 8, 4);
    else if (heads == 6 && tile_k == 16 && coalesce == 1) V9_LAUNCH(6, 16, 1);
    else if (heads == 6 && tile_k == 16 && coalesce == 2) V9_LAUNCH(6, 16, 2);
    else if (heads == 6 && tile_k == 16 && coalesce == 4) V9_LAUNCH(6, 16, 4);
    else if (heads == 3 && tile_k == 8  && coalesce == 1) V9_LAUNCH(3, 8, 1);
    else if (heads == 3 && tile_k == 8  && coalesce == 2) V9_LAUNCH(3, 8, 2);
    else if (heads == 3 && tile_k == 8  && coalesce == 4) V9_LAUNCH(3, 8, 4);
    else if (heads == 3 && tile_k == 16 && coalesce == 1) V9_LAUNCH(3, 16, 1);
    else if (heads == 3 && tile_k == 16 && coalesce == 2) V9_LAUNCH(3, 16, 2);
    else if (heads == 3 && tile_k == 16 && coalesce == 4) V9_LAUNCH(3, 16, 4);
    else if (heads == 2 && tile_k == 8  && coalesce == 1) V9_LAUNCH(2, 8, 1);
    else if (heads == 2 && tile_k == 8  && coalesce == 2) V9_LAUNCH(2, 8, 2);
    else if (heads == 2 && tile_k == 8  && coalesce == 4) V9_LAUNCH(2, 8, 4);
    else if (heads == 2 && tile_k == 16 && coalesce == 1) V9_LAUNCH(2, 16, 1);
    else if (heads == 2 && tile_k == 16 && coalesce == 2) V9_LAUNCH(2, 16, 2);
    else if (heads == 2 && tile_k == 16 && coalesce == 4) V9_LAUNCH(2, 16, 4);
    else if (heads == 1 && tile_k == 8  && coalesce == 1) V9_LAUNCH(1, 8, 1);
    else if (heads == 1 && tile_k == 8  && coalesce == 2) V9_LAUNCH(1, 8, 2);
    else if (heads == 1 && tile_k == 8  && coalesce == 4) V9_LAUNCH(1, 8, 4);
    else if (heads == 1 && tile_k == 16 && coalesce == 1) V9_LAUNCH(1, 16, 1);
    else if (heads == 1 && tile_k == 16 && coalesce == 2) V9_LAUNCH(1, 16, 2);
    else if (heads == 1 && tile_k == 16 && coalesce == 4) V9_LAUNCH(1, 16, 4);
    else ASSERT(false, "flash_decoding_v9: unsupported (heads, tile_k, coalesce)");

#undef V9_LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void flash_decoding_v9_dispatch(
    std::byte *attn_val, std::byte *attn_acc, std::byte *attn_sum, std::byte *attn_max,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *cut_idx, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    size_t tot_block_num, size_t tot_task_num,
    float scale, size_t nh, size_t dv, size_t d, size_t nkvh,
    int heads, int tile_k, int coalesce)
{
    ASSERT(d == dv, "flash_decoding_v9: d must equal dv.");
#define V9_IMPL(HD)                                                                   \
    flash_decoding_v9_impl<T, HD>(attn_val, attn_acc, attn_sum, attn_max,            \
        q, k_cache, v_cache, block_ids, cut_idx, tot_len,                             \
        token_num, batch_size, max_block_num, tot_block_num, tot_task_num,            \
        scale, nh, nkvh, heads, tile_k, coalesce)
    switch (d) {
    case 128: V9_IMPL(128); break;
    case 64:  V9_IMPL(64);  break;
    case 32:  V9_IMPL(32);  break;
    case 16:  V9_IMPL(16);  break;
    default:  ASSERT(false, "flash_decoding_v9: unsupported head_dim.");
    }
#undef V9_IMPL
}

// ── C wrapper ───────────────────────────────────────────────────────
extern "C" __export void llaisysFlashDecodingV9(
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

    std::vector<int64_t> h_bids(batch_size * max_block_num);
    cudaMemcpy(h_bids.data(), block_ids, batch_size * max_block_num * sizeof(int64_t), cudaMemcpyDeviceToHost);
    size_t ttn = 0; int64_t p = 0; size_t co = static_cast<size_t>(coalesce > 0 ? coalesce : 1);
    for (size_t i = 0; i < batch_size; ++i)
        { int64_t c = h_bids[i * max_block_num]; ttn += static_cast<size_t>((static_cast<uint64_t>(c - p) + co - 1) / co); p = c; }

    switch (dtype) {
    case 0: flash_decoding_v9_dispatch<float>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, ttn, scale, nh, dv, d, nkvh, heads, tile_k, coalesce); break;
    case 1: flash_decoding_v9_dispatch<llaisys::fp16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, ttn, scale, nh, dv, d, nkvh, heads, tile_k, coalesce); break;
    case 2: flash_decoding_v9_dispatch<llaisys::bf16_t>(ba, bac, bas, bam, bq, bkc, bvc, bb, bc, bt,
        token_num, batch_size, max_block_num, tot_block_num, ttn, scale, nh, dv, d, nkvh, heads, tile_k, coalesce); break;
    default: ASSERT(false, "flash_decoding_v9: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
