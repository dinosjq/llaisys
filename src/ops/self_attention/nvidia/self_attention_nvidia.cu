#include "self_attention_nvidia.cuh"
#include "utils.hpp"
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

// 0 1 2 3 4 5 ... 127   =>  0 32 64 96 1 33 ... 127
// physical addr => logical addr
__device__ __forceinline__ size_t hash(size_t idx) {
    return ((idx & 3) << 5) + (idx >> 2);
}

// logical addr => physical addr
__device__ __forceinline__ size_t in_hash(size_t idx) {
    return ((idx & 31) << 2) + (idx >> 5);
}

static constexpr size_t BLOCK_M = 8;
static constexpr size_t block_dim = 256;

// 针对模型特化的 flash_attention
template <typename T>
__global__ void flash_attention_kernel(T *__restrict__ _attn, const T *__restrict__ _Q, const T *__restrict__ _K, const T *__restrict__ _V,
                                       const float _scale, const size_t _seqlen, const size_t _nhead, const size_t _dv, const size_t _d,
                                       const size_t _totlen, const size_t _nkvhead) {

    extern __shared__ float smem[];
    float *_s_acc = smem;                 // (BLOCK_M X dv) 计算的暂存值
    float *_s_l = _s_acc + BLOCK_M * _dv; // (BLOCK_M X 1)  分母和
    float *_s_m = _s_l + BLOCK_M;         // (BLOCK_M X 1)  最大值
    float *_s_alpha = _s_m + BLOCK_M;     // (BLOCK_M X 1)
    float *_s_beta = _s_alpha + BLOCK_M;  // (BLOCK_M X 1)
    float *_s_q = _s_beta + BLOCK_M;      // (BLOCK_M X d)
    float *_s_k = _s_q + BLOCK_M * _d;    // (BLOCK_M X d)
    float *_s_v = _s_k + BLOCK_M * _d;    // (BLOCK_M X dv)

    const size_t q = blockIdx.x * BLOCK_M;
    const size_t nh = blockIdx.y;
    const size_t kv_repeat = (_nkvhead != 0 && _nhead % _nkvhead == 0) ? (_nhead / _nkvhead) : 0;
    const size_t nkvh = (kv_repeat > 0) ? (nh / kv_repeat) : (nh % _nkvhead);
    const size_t extra = (_totlen > _seqlen) ? (_totlen - _seqlen) : 0;

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    const size_t qi = q + warp_id;
    const size_t limit = min(qi + extra, _totlen - 1);

    const T *_q = _Q + (qi * _nhead + nh) * _d;
    float *s_acc = _s_acc + warp_id * _dv;
    float *s_q = _s_q + warp_id * _d;

    // init acc / l / m
    {
        // naive: warp init
        for (size_t i = lane_id; i < _dv; i += 32) {
            s_acc[i] = 0.0f;
        }
        if (tid < 8) {
            _s_l[tid] = 0.0f;
            _s_m[tid] = -INFINITY;
        }
    }

    // load q
    {
        // naive: warp init
        if (qi < _seqlen) {
            // vec load: assert _d = 128
            for (size_t col = lane_id << 2; col < _d; col += 128) {
                const float4 flo4 = llaisys::utils::nvidia::load_4d(_q + col);
                // physical: [col + 0, col + 1, col + 2, col + 3]
                s_q[hash(col | 0)] = flo4.x;
                s_q[hash(col | 1)] = flo4.y;
                s_q[hash(col | 2)] = flo4.z;
                s_q[hash(col | 3)] = flo4.w;
            }
        }
    }

    __syncwarp();

    for (size_t k = 0; k < _totlen; ++k) {
        // load k / v
        if (k % BLOCK_M == 0) {
            __syncthreads();
            const size_t offset = k + warp_id;
            if (offset < _totlen) {
                const T *_k = _K + (offset * _nkvhead + nkvh) * _d;
                const T *_v = _V + (offset * _nkvhead + nkvh) * _dv;

                // special realize: _d = _dv = 128
                float *s_k = _s_k + warp_id * _d;
                float *s_v = _s_v + warp_id * _dv;

                // vec load: warp init
                for (size_t col = lane_id << 2; col < _d; col += 128) {
                    const float4 flo4 = llaisys::utils::nvidia::load_4d(_k + col);
                    // physical: [col + 0, col + 1, col + 2, col + 3]
                    s_k[hash(col | 0)] = flo4.x;
                    s_k[hash(col | 1)] = flo4.y;
                    s_k[hash(col | 2)] = flo4.z;
                    s_k[hash(col | 3)] = flo4.w;
                }

                for (size_t col = lane_id << 2; col < _dv; col += 128) {
                    const float4 flo4 = llaisys::utils::nvidia::load_4d(_v + col);
                    s_v[hash(col | 0)] = flo4.x;
                    s_v[hash(col | 1)] = flo4.y;
                    s_v[hash(col | 2)] = flo4.z;
                    s_v[hash(col | 3)] = flo4.w;
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
        const size_t offset = (qi * _nhead + nh) * _dv;

        for (size_t i = lane_id; i < _dv; i += 32) {
            const float out = s_acc[i] * inv;
            const size_t idx = offset + in_hash(i);

            _attn[idx] = llaisys::utils::nvidia::cast<T>(out);
        }
    }
}

template <typename T>
void self_attention_launch(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, const float &scale,
                           const size_t &seqlen, const size_t &nhead, const size_t &dv, const size_t &d, const size_t &total_len, const size_t &nkvhead) {
    auto *d_attn = reinterpret_cast<T *>(attn_val);
    const auto *d_q = reinterpret_cast<const T *>(q);
    const auto *d_k = reinterpret_cast<const T *>(k);
    const auto *d_v = reinterpret_cast<const T *>(v);

    dim3 blockDim(256);
    dim3 gridDim((seqlen + 7) >> 3, nhead);

    const size_t smem_bytes = sizeof(float) * (BLOCK_M * (d + dv + 4 + d + dv));

    flash_attention_kernel<<<gridDim, blockDim, smem_bytes>>>(d_attn, d_q, d_k, d_v, scale, seqlen, nhead, dv, d, total_len, nkvhead);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, const float &scale, llaisysDataType_t type,
                    const size_t &seqlen, const size_t &nhead, const size_t &dv, const size_t &d, const size_t &total_len, const size_t &nkvhead) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_launch<float>(attn_val, q, k, v, scale, seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_BF16:
        return self_attention_launch<llaisys::bf16_t>(attn_val, q, k, v, scale, seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_F16:
        return self_attention_launch<llaisys::fp16_t>(attn_val, q, k, v, scale, seqlen, nhead, dv, d, total_len, nkvhead);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia