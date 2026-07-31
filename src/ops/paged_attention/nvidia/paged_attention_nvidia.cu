#include "../../../utils.hpp"
#include "paged_attention_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

static constexpr size_t WARP_SIZE = 32;
static constexpr size_t BLOCK_DIM = 256;
static constexpr size_t BLOCK_M = 8;
static constexpr size_t COMPUTE_WARPS = 4;
static constexpr size_t KV_TILE = 4;
static constexpr size_t HEAD_DIM_MAX = 128;
static constexpr size_t VEC_SIZE = 4;
static constexpr unsigned int FULL_MASK = 0xFFFFFFFFU;

__device__ __forceinline__ float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ __forceinline__ float4 zero_float4() {
    return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

__device__ __forceinline__ float dot_float4(const float4 &lhs, const float4 &rhs) {
    return lhs.x * rhs.x + lhs.y * rhs.y + lhs.z * rhs.z + lhs.w * rhs.w;
}

template <typename T>
__device__ __forceinline__ void load_paged_kv_row(
    float *__restrict__ s_k,
    float *__restrict__ s_v,
    const T *__restrict__ k_cache,
    const T *__restrict__ v_cache,
    const int64_t *__restrict__ block_ids,
    const size_t logical_token,
    const size_t kv_end,
    const size_t buffer,
    const size_t row,
    const size_t token_num,
    const size_t nkvhead,
    const size_t nkvh,
    const size_t d,
    const size_t dv) {
    if (logical_token >= kv_end) {
        return;
    }

    const int64_t block_id = block_ids[logical_token / token_num + 1];
    const size_t block_token = logical_token % token_num;
    const T *k_src = k_cache + ((block_id * token_num + block_token) * nkvhead + nkvh) * d;
    const T *v_src = v_cache + ((block_id * token_num + block_token) * nkvhead + nkvh) * dv;
    float *k_dst = s_k + (buffer * KV_TILE + row) * d;
    float *v_dst = s_v + (buffer * KV_TILE + row) * dv;
    const size_t col = (threadIdx.x & (WARP_SIZE - 1)) * VEC_SIZE;

    if (col < d) {
        llaisys::utils::nvidia::copy_4d(k_src + col, k_dst + col);
    }
    if (col < dv) {
        llaisys::utils::nvidia::copy_4d(v_src + col, v_dst + col);
    }
}

// 自定义 paged_attention 以 flash_attention_v2 为基础
template <typename T>
__global__ void paged_attention_kernel(T *__restrict__ _attn, 
                                       const T *__restrict__ _Q, 
                                       const T *__restrict__ _K_cache, 
                                       const T *__restrict__ _V_cache, 
                                       const int64_t *_block_ids,
                                       const int64_t *_cut_idx,
                                       const int64_t *_tot_len,
                                       const size_t _token_num,
                                       const size_t _max_block_num,
                                       const float _scale, 
                                       const size_t _nhead, 
                                       const size_t _dv, 
                                       const size_t _d,
                                       const size_t _nkvhead) 
{
    extern __shared__ float4 smem_vec[];
    float *smem = reinterpret_cast<float *>(smem_vec);
    float *s_k = smem;
    float *s_v = s_k + 2 * KV_TILE * _d;

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid / WARP_SIZE;
    const size_t lane_id = tid & (WARP_SIZE - 1);
    const size_t batch_id = blockIdx.z;
    const bool consumer_warp = warp_id < COMPUTE_WARPS;
    const bool producer_warp = !consumer_warp;

    const int64_t q_start = _cut_idx[batch_id];
    const int64_t seq_len_i64 = _cut_idx[batch_id + 1] - q_start;
    const int64_t total_len_i64 = _tot_len[batch_id];
    const int64_t prefix_len_i64 = total_len_i64 - seq_len_i64;
    const size_t q_base = blockIdx.x * BLOCK_M;

    // Every predicate here is block-uniform, so returning before the first
    // block-wide barrier is safe.
    if (q_start < 0 || seq_len_i64 <= 0 || total_len_i64 <= 0
        || prefix_len_i64 < 0
        || q_base >= static_cast<size_t>(seq_len_i64)) {
        return;
    }

    const size_t seq_len = static_cast<size_t>(seq_len_i64);
    const size_t prefix_len = static_cast<size_t>(prefix_len_i64);
    const size_t valid_q_end = (q_base + BLOCK_M < seq_len) ? q_base + BLOCK_M : seq_len;
    const size_t kv_end = prefix_len + valid_q_end;
    const size_t nh = blockIdx.y;
    const size_t kv_rep = _nhead / _nkvhead;
    const size_t nkvh = nh / kv_rep;
    const int64_t *batch_block_ids =
        _block_ids + batch_id * _max_block_num;

    float4 q_frag[2] = {zero_float4(), zero_float4()};
    float4 acc_frag[2] = {zero_float4(), zero_float4()};
    float running_max[2] = {-INFINITY, -INFINITY};
    float running_sum[2] = {0.0f, 0.0f};
    size_t query_index[2] = {0, 0};
    bool query_valid[2] = {false, false};
    const size_t q_col = lane_id * VEC_SIZE;
    const size_t v_col = lane_id * VEC_SIZE;

    if (consumer_warp) {
        query_index[0] = q_base + warp_id;
        query_index[1] = query_index[0] + COMPUTE_WARPS;
        query_valid[0] = query_index[0] < seq_len;
        query_valid[1] = query_index[1] < seq_len;

#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            if (query_valid[slot] && q_col < _d) {
                const size_t global_query = static_cast<size_t>(q_start) + query_index[slot];
                const T *q_src = _Q + (global_query * _nhead + nh) * _d + q_col;
                q_frag[slot] = llaisys::utils::nvidia::load_4d(q_src);
            }
        }
    }

    const size_t producer_row = warp_id - COMPUTE_WARPS;
    if (producer_warp) {
        load_paged_kv_row(
            s_k, s_v, _K_cache, _V_cache, batch_block_ids,
            producer_row, kv_end, 0, producer_row, _token_num,
            _nkvhead, nkvh, _d, _dv);
    }

    // Publish the initial KV tile before any consumer reads it.
    __syncthreads();

    size_t buffer = 0;
    for (size_t tile_base = 0; tile_base < kv_end;
         tile_base += KV_TILE) {
        if (producer_warp) {
            const size_t next_token = tile_base + KV_TILE + producer_row;
            load_paged_kv_row(
                s_k, s_v, _K_cache, _V_cache, batch_block_ids,
                next_token, kv_end, buffer ^ 1, producer_row,
                _token_num, _nkvhead, nkvh, _d, _dv);
        }

        if (consumer_warp) {
#pragma unroll
            for (size_t row = 0; row < KV_TILE; ++row) {
                const size_t logical_token = tile_base + row;
                if (logical_token >= kv_end) {
                    break;
                }

                const float *k_row = s_k + (buffer * KV_TILE + row) * _d;
                const float *v_row = s_v + (buffer * KV_TILE + row) * _dv;
                float4 k_frag = zero_float4();
                float4 v_frag = zero_float4();
                if (q_col < _d) {
                    k_frag = *reinterpret_cast<const float4 *>(k_row + q_col);
                }
                if (v_col < _dv) {
                    v_frag = *reinterpret_cast<const float4 *>(v_row + v_col);
                }

#pragma unroll
                for (int slot = 0; slot < 2; ++slot) {
                    if (!query_valid[slot]) {
                        continue;
                    }
                    const size_t query_kv_end = prefix_len + query_index[slot] + 1;
                    if (logical_token >= query_kv_end) {
                        continue;
                    }

                    float dot = dot_float4(q_frag[slot], k_frag);
                    dot = warp_reduce_sum(dot);
                    float alpha = 1.0f;
                    float beta = 0.0f;

                    if (lane_id == 0) {
                        const float score = dot * _scale;
                        const float new_max = fmaxf(running_max[slot], score);
                        alpha = expf(running_max[slot] - new_max);
                        beta = expf(score - new_max);
                        running_sum[slot] = running_sum[slot] * alpha + beta;
                        running_max[slot] = new_max;
                    }

                    alpha = __shfl_sync(FULL_MASK, alpha, 0);
                    beta = __shfl_sync(FULL_MASK, beta, 0);
                    if (v_col < _dv) {
                        acc_frag[slot].x = acc_frag[slot].x * alpha + v_frag.x * beta;
                        acc_frag[slot].y = acc_frag[slot].y * alpha + v_frag.y * beta;
                        acc_frag[slot].z = acc_frag[slot].z * alpha + v_frag.z * beta;
                        acc_frag[slot].w = acc_frag[slot].w * alpha + v_frag.w * beta;
                    }
                }
            }
        }

        // Producers publish the next buffer and consumers finish reading the
        // current one before their roles swap.
        __syncthreads();
        buffer ^= 1;
    }

    if (consumer_warp) {
#pragma unroll
        for (int slot = 0; slot < 2; ++slot) {
            if (!query_valid[slot]) {
                continue;
            }

            float inv_sum = 0.0f;
            if (lane_id == 0 && running_sum[slot] > 0.0f) {
                inv_sum = 1.0f / running_sum[slot];
            }
            inv_sum = __shfl_sync(FULL_MASK, inv_sum, 0);

            if (v_col < _dv) {
                const size_t global_query = static_cast<size_t>(q_start) + query_index[slot];
                T *out = _attn + (global_query * _nhead + nh) * _dv + v_col;
                out[0] = llaisys::utils::nvidia::cast<T>(acc_frag[slot].x * inv_sum);
                out[1] = llaisys::utils::nvidia::cast<T>(acc_frag[slot].y * inv_sum);
                out[2] = llaisys::utils::nvidia::cast<T>(acc_frag[slot].z * inv_sum);
                out[3] = llaisys::utils::nvidia::cast<T>(acc_frag[slot].w * inv_sum);
            }
        }
    }
}

template <typename T>
void paged_attention_launch(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                            const std::byte *q,        // q (tot_seqlen, nh, d)
                            const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                            const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                            const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, token_num, nkvh, d | dv) 
                            const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                            const std::byte *tot_len,  // 已计算kv总长
                            const size_t token_num,    // 每个块的token数 (批次无关)
                            const size_t batch_size,   // 批次大小 (批次有关)
                            const size_t max_block_num,// 块表最多块数 (批次无关)
                            const size_t max_seq_len,  // 最大序列长度 用于分块 (批次无关)
                            const float &scale,        // 缩放因子
                            const size_t &nh,          // 其他参数 ...
                            const size_t &dv, 
                            const size_t &d, 
                            const size_t &nkvh)
{
    if (d == 0 || dv == 0
        || d > HEAD_DIM_MAX || dv > HEAD_DIM_MAX
        || d % 32 != 0 || dv % 32 != 0
        || token_num == 0 || nkvh == 0 || nh % nkvh != 0) {
        throw std::invalid_argument(
            "paged_attention requires d,dv in {32,64,96,128}, "
            "token_num > 0, nkvh > 0, and nh divisible by nkvh");
    }
    if (batch_size == 0 || nh == 0 || max_seq_len == 0) {
        return;
    }

    auto *d_attn = reinterpret_cast<T *>(attn_val);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const auto *d_k_cache = reinterpret_cast<const T *>(k_cache);
    const auto *d_v_cache = reinterpret_cast<const T *>(v_cache);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(
        (max_seq_len + BLOCK_M - 1) / BLOCK_M, nh, batch_size);

    const size_t smem_bytes =
        sizeof(float) * 2 * KV_TILE * (d + dv);

    paged_attention_kernel<<<gridDim, blockDim, smem_bytes>>>(
        d_attn, d_q, d_k_cache, d_v_cache, d_block_ids, d_cut_idx,
        d_tot_len, token_num, max_block_num, scale, nh, dv, d, nkvh);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void paged_attention(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                     const std::byte *q,        // q (tot_seqlen, nh, d)
                     const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                     const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                     const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, token_num, nkvh, d | dv) 
                     const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                     const std::byte *tot_len,  // 已计算kv总长
                     const size_t token_num,    // 每个块的token数
                     const size_t batch_size,   // 批次大小
                     const size_t max_block_num,// 块表最多块数
                     const size_t max_seq_len,  // 最大序列长度 用于分块
                     llaisysDataType_t dtype,   // 权重数据类型
                     const float &scale,        // 缩放因子
                     const size_t &nh,          // 其他参数 ...
                     const size_t &dv, 
                     const size_t &d, 
                     const size_t &nkvh)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return paged_attention_launch<float>(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_BF16:
        return paged_attention_launch<llaisys::bf16_t>(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    case LLAISYS_DTYPE_F16:
        return paged_attention_launch<llaisys::fp16_t>(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, token_num,
                                             batch_size, max_block_num, max_seq_len, scale, nh, dv, d, nkvh);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia