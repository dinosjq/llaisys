#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "flash_decoding_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

// ── v8: block-per-seq×kv_head 单 kernel (FlashInfer 架构) ────────
// 每 block 服务一个 (seq, kv_head), 串行遍历该 seq 全部 KV token,
// online softmax 直接累积, 最终写出 attn_val. 无 reduce kernel.
//
// 线程布局 (仿 FlashInfer):
//   * grid = (batch, nkvh)
//   * blockDim = (16, rep), rep = nh/nkvh  (H6 → 96 threads)
//   * threadIdx.x (0-15): head_dim 的 [x*8, (x+1)*8) 段
//   * threadIdx.y (0..rep-1): Q head (共享当前 kv_head)
//   * q 常驻寄存器 8 元素, acc 8 元素/thread
//   * 点积 8 FMA + shuffle width=16 归约
//
// 本版: 直接读全局 KV (无 smem pipeline), 验证正确性 + 性能基线.

__device__ __forceinline__ float subwarp_reduce_16(float val) {
#pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1)
        val += ::__shfl_xor_sync(0xFFFFFFFF, val, offset, 16);
    return val;
}

static constexpr size_t ELEMS_PER_THREAD_V8 = 8;  // HD=128 / 16 threads

template <typename T, size_t HD, size_t REP>
__global__ void flash_decoding_v8_kernel(
    T *__restrict__ _attn,
    const T *__restrict__ _q, const T *__restrict__ _k_cache, const T *__restrict__ _v_cache,
    const int64_t *__restrict__ _block_ids, const int64_t *__restrict__ _tot_len,
    const size_t _token_num, const size_t _max_block_num,
    const float _scale, const size_t _nkvh)
{
    const size_t xseg   = threadIdx.x;   // 0..15, head_dim 段
    const size_t head   = threadIdx.y;   // 0..REP-1, Q head (相对 kv_head)
    const size_t tid    = threadIdx.y * 16 + threadIdx.x;

    const size_t seq_id   = blockIdx.x;   // batch 内 seq
    const size_t kv_head  = blockIdx.y;   // kv head

    // Q head 全局索引 = kv_head * REP + head
    const size_t global_head = kv_head * REP + head;

    // ── Q 常驻寄存器 ────────────────────────────────────────────
    const T *qh = _q + (seq_id * (_nkvh * REP) + global_head) * HD + xseg * ELEMS_PER_THREAD_V8;
    float q_reg[ELEMS_PER_THREAD_V8];
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD_V8; ++i)
        q_reg[i] = llaisys::utils::nvidia::cast<float>(qh[i]);

    // ── 页表遍历 ────────────────────────────────────────────────
    const int64_t *bids = _block_ids + seq_id * _max_block_num;
    const size_t totlen = static_cast<size_t>(_tot_len[seq_id]);
    const size_t limit  = totlen - 1;
    // 首列是累计前缀和, 本 seq 块数需差分
    const int64_t prev_cum = (seq_id > 0)
        ? _block_ids[(seq_id - 1) * _max_block_num] : 0;
    const size_t bn = static_cast<size_t>(bids[0] - prev_cum);

    float acc[ELEMS_PER_THREAD_V8];
    float reg_sum = 0.0f, reg_max = -INFINITY;
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD_V8; ++i)
        acc[i] = 0.0f;

    // 遍历该 seq 的所有 KV block × token
    size_t global_tok = 0;  // seq 内绝对 token 位置
    for (size_t bi = 0; bi < bn; ++bi) {
        const int64_t block_id = bids[bi + 1];
        const T *k_base = _k_cache + (static_cast<size_t>(block_id) * _token_num * _nkvh + kv_head) * HD;
        const T *v_base = _v_cache + (static_cast<size_t>(block_id) * _token_num * _nkvh + kv_head) * HD;
        for (size_t tok = 0; tok < _token_num; ++tok, ++global_tok) {
            if (global_tok > limit) break;
            const T *kp = k_base + tok * _nkvh * HD + xseg * ELEMS_PER_THREAD_V8;
            const T *vp = v_base + tok * _nkvh * HD + xseg * ELEMS_PER_THREAD_V8;
            float dot = 0.0f;
#pragma unroll
            for (size_t i = 0; i < ELEMS_PER_THREAD_V8; ++i)
                dot += q_reg[i] * llaisys::utils::nvidia::cast<float>(kp[i]);
            dot = subwarp_reduce_16(dot) * _scale;
            float mn = fmaxf(reg_max, dot);
            float a  = __expf(reg_max - mn);
            float b2 = __expf(dot - mn);
            reg_max = mn;
            reg_sum = a * reg_sum + b2;
#pragma unroll
            for (size_t i = 0; i < ELEMS_PER_THREAD_V8; ++i)
                acc[i] = a * acc[i] + b2 * llaisys::utils::nvidia::cast<float>(vp[i]);
        }
    }

    // ── 写最终结果 ──────────────────────────────────────────────
    float den = (reg_sum > 0.0f) ? (1.0f / reg_sum) : 1.0f;
    T *out = _attn + (seq_id * (_nkvh * REP) + global_head) * HD + xseg * ELEMS_PER_THREAD_V8;
#pragma unroll
    for (size_t i = 0; i < ELEMS_PER_THREAD_V8; ++i)
        out[i] = llaisys::utils::nvidia::cast<T>(acc[i] * den);
}

// ── Host launch ────────────────────────────────────────────────────
template <typename T, size_t HD, size_t REP>
void launch_v8(std::byte *attn_val,
               const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
               const std::byte *block_ids, const std::byte *tot_len,
               size_t token_num, size_t batch_size, size_t max_block_num,
               float scale, size_t nkvh)
{
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    const auto *dq = reinterpret_cast<const T *>(q);
    const auto *dk = reinterpret_cast<const T *>(k_cache);
    const auto *dv = reinterpret_cast<const T *>(v_cache);
    const auto *db = reinterpret_cast<const int64_t *>(block_ids);
    const auto *dt = reinterpret_cast<const int64_t *>(tot_len);

    dim3 grid0(batch_size, nkvh);
    dim3 blockDim(16, REP);

    auto *kernel = flash_decoding_v8_kernel<T, HD, REP>;
    kernel<<<grid0, blockDim>>>(d_attn, dq, dk, dv, db, dt, token_num, max_block_num, scale, nkvh);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
static void flash_decoding_v8_dispatch(
    std::byte *attn_val,
    const std::byte *q, const std::byte *k_cache, const std::byte *v_cache,
    const std::byte *block_ids, const std::byte *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    float scale, size_t nkvh, size_t d)
{
#define V8_IMPL(HD)                                                                    \
    launch_v8<T, HD, 6>(attn_val, q, k_cache, v_cache, block_ids, tot_len,             \
                        token_num, batch_size, max_block_num, scale, nkvh)             // REP=6 (nh=12/nkvh=2)
    switch (d) {
    case 128: V8_IMPL(128); break;
    case 64:  V8_IMPL(64);  break;
    default:  ASSERT(false, "flash_decoding_v8: unsupported head_dim.");
    }
#undef V8_IMPL
}

// ── C wrapper (block_ids 布局: 首列累计块数) ──────────────────────
extern "C" __export void llaisysFlashDecodingV8(
    void *attn_val, void *q, void *k_cache, void *v_cache,
    void *block_ids, void *tot_len,
    size_t token_num, size_t batch_size, size_t max_block_num,
    size_t max_seq_len, int dtype, float scale, size_t nh, size_t dv, size_t d, size_t nkvh)
{
    auto *ba  = reinterpret_cast<std::byte *>(attn_val);
    auto *bq  = reinterpret_cast<const std::byte *>(q);
    auto *bkc = reinterpret_cast<const std::byte *>(k_cache);
    auto *bvc = reinterpret_cast<const std::byte *>(v_cache);
    auto *bb  = reinterpret_cast<const std::byte *>(block_ids);
    auto *bt  = reinterpret_cast<const std::byte *>(tot_len);

    switch (dtype) {
    case 0: flash_decoding_v8_dispatch<float>(ba, bq, bkc, bvc, bb, bt,
        token_num, batch_size, max_block_num, scale, nkvh, d); break;
    case 1: flash_decoding_v8_dispatch<llaisys::fp16_t>(ba, bq, bkc, bvc, bb, bt,
        token_num, batch_size, max_block_num, scale, nkvh, d); break;
    case 2: flash_decoding_v8_dispatch<llaisys::bf16_t>(ba, bq, bkc, bvc, bb, bt,
        token_num, batch_size, max_block_num, scale, nkvh, d); break;
    default: ASSERT(false, "flash_decoding_v8: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
