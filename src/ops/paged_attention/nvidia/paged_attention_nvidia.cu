#include "../../../utils.hpp"
#include "paged_attention_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

__device__ __forceinline__ float warp_reduce(float val) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

static constexpr size_t BLOCK_M = 8;
static constexpr size_t block_dim = 256;

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

    extern __shared__ float smem[];
    float *_s_acc = smem;                 // (BLOCK_M X dv) 计算的暂存值
    float *_s_l = _s_acc + BLOCK_M * _dv; // (BLOCK_M X 1)  分母和
    float *_s_m = _s_l + BLOCK_M;         // (BLOCK_M X 1)  最大值
    float *_s_alpha = _s_m + BLOCK_M;     // (BLOCK_M X 1)
    float *_s_beta = _s_alpha + BLOCK_M;  // (BLOCK_M X 1)
    float *_s_q = _s_beta + BLOCK_M;      // (BLOCK_M X d)
    float *_s_k = _s_q + BLOCK_M * _d;    // (BLOCK_M X d)
    float *_s_v = _s_k + BLOCK_M * _d;    // (BLOCK_M X dv)

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;
    const size_t batch_id = blockIdx.z;

    const int64_t *batch_block_ids = _block_ids + batch_id * _max_block_num;
    const size_t _seqlen = _cut_idx[batch_id + 1] - _cut_idx[batch_id];
    const size_t _totlen = _tot_len[batch_id];

    const size_t q = blockIdx.x * BLOCK_M;
    const size_t nh = blockIdx.y;
    const size_t kv_rep = _nhead / _nkvhead;
    const size_t nkvh = nh / kv_rep;
    const size_t extra = _totlen - _seqlen;

    const size_t qi = q + warp_id;
    const size_t limit = min(qi + extra, _totlen - 1);

    const T *_q = _Q + ((qi + _cut_idx[batch_id]) * _nhead + nh) * _d;
    float *s_acc = _s_acc + warp_id * _dv;
    float *s_q = _s_q + warp_id * _d;

    // init acc / l / m
    {
        // naive: warp init
        for (size_t i = lane_id; i < _dv; i += 32) {
            s_acc[i] = 0.0f;
        }
        if (lane_id == 0) {
            _s_l[warp_id] = 0.0f;
            _s_m[warp_id] = -INFINITY;
        }
    }

    // load q
    {
        // naive: warp init
        if (qi < _seqlen) {
            // vec load: assert _d = 128
            for (size_t col = lane_id << 2; col < _d; col += 128) {
                llaisys::utils::nvidia::copy_4d(_q + col, s_q + col);
            }
        }
    }

    __syncwarp();

    for (size_t k = 0; k < _totlen; ++k) {
        // load k / v
        if (k % BLOCK_M == 0) {
            __syncthreads();

            size_t token_id = k + warp_id;

            if(token_id < _totlen){
                const int64_t block_id = batch_block_ids[token_id / _token_num + 1]; // 这里得 +1 每行第一个数据为该 seq 的块表大小
                token_id %= _token_num;

                // global memory
                const T *_k = _K_cache + ((block_id * _token_num + token_id) * _nkvhead + nkvh) * _d;
                const T *_v = _V_cache + ((block_id * _token_num + token_id) * _nkvhead + nkvh) * _dv;

                // shared memory
                float *s_k = _s_k + warp_id * _d;
                float *s_v = _s_v + warp_id * _dv;

                // vec load: warp init
                for (size_t col = lane_id << 2; col < _d; col += 128) {
                    llaisys::utils::nvidia::copy_4d(_k + col, s_k + col);
                }

                for (size_t col = lane_id << 2; col < _dv; col += 128) {
                    llaisys::utils::nvidia::copy_4d(_v + col, s_v + col);
                }
            }
            __syncthreads();
        }

        const float *s_k = _s_k + (k & 7) * _d;
        const float *s_v = _s_v + (k & 7) * _dv;

        // online softmax
        {
            // naive: warp compute
            if (qi < _seqlen && k <= limit) {
                float dot = 0.0f;
                for (size_t i = lane_id; i < _d; i += 32) {
                    dot += s_q[i] * s_k[i];
                }
                dot = warp_reduce(dot);

                if (lane_id == 0) {
                    const float s = dot * _scale;
                    const float m_old = _s_m[warp_id];
                    const float m_new = fmaxf(m_old, s);
                    const float alpha = expf(m_old - m_new);
                    const float beta = expf(s - m_new);

                    _s_m[warp_id] = m_new;
                    _s_l[warp_id] = _s_l[warp_id] * alpha + beta;
                    _s_alpha[warp_id] = alpha;
                    _s_beta[warp_id] = beta;
                }
            } else {
                _s_alpha[warp_id] = 1.0f;
                _s_beta[warp_id] = 0.0f;
            }

            __syncwarp();

            // naive: warp update
            if (qi < _seqlen && k <= limit) {
                const float alpha = _s_alpha[warp_id];
                const float beta = _s_beta[warp_id];

                for (size_t i = lane_id; i < _dv; i += 32) {
                    s_acc[i] = s_acc[i] * alpha + s_v[i] * beta;
                }
            }

            __syncwarp();
        }
    }

    // write out
    {
        // naive: warp load back
        if (qi >= _seqlen) {
            return;
        }
        const float den = _s_l[warp_id];
        const float inv = (den > 0.0f) ? 1.0f / den : 0.0f;
        const size_t offset = ((qi + _cut_idx[batch_id]) * _nhead + nh) * _dv;

        for (size_t i = lane_id; i < _dv; i += 32) {
            const float out = s_acc[i] * inv;
            const size_t idx = offset + i;

            _attn[idx] = llaisys::utils::nvidia::cast<T>(out);
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
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const auto *d_k_cache = reinterpret_cast<const T *>(k_cache);
    const auto *d_v_cache = reinterpret_cast<const T *>(v_cache);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_tot_len = reinterpret_cast<const int64_t *>(tot_len);

    dim3 blockDim(256);
    dim3 gridDim((max_seq_len + 7) >> 3, nh, batch_size);

    const size_t smem_bytes = sizeof(float) * (BLOCK_M * (d + dv + 4 + d + dv));

    paged_attention_kernel<<<gridDim, blockDim, smem_bytes>>>(d_attn, d_q, d_k_cache, d_v_cache, d_block_ids, d_cut_idx, d_tot_len,
                                                                 token_num, max_block_num, scale, nh, dv, d, nkvh);

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
