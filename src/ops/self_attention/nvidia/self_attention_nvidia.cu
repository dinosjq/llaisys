#include "self_attention_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

constexpr static int BLOCK_M = 32;

template <typename T>
__global__ void flash_attention_kernel(T *attn, const T *Q, const T *K, const T *V, const float scale, const size_t seqlen,
                                       const size_t nhead, const size_t dv, const size_t d, const size_t totlen, const size_t nkvhead) {
    const size_t q = blockIdx.x * BLOCK_M;
    const size_t nh = blockIdx.y;
    const size_t tid = threadIdx.x;

    const size_t kv_repeat = (nkvhead != 0 && nhead % nkvhead == 0) ? (nhead / nkvhead) : 0;

    const size_t nkvh = (kv_repeat > 0) ? (nh / kv_repeat) : (nh % nkvhead);

    extern __shared__ float smem[];
    float *s_acc = smem;                                    // (BLOCK_M X dv) 计算的暂存值
    float *s_l = s_acc + static_cast<size_t>(BLOCK_M) * dv; // (BLOCK_M X 1) 分母和
    float *s_m = s_l + BLOCK_M;                             // (BLOCK_M X 1) 最大值
    float *s_alpha = s_m + BLOCK_M;                         // (BLOCK_M X 1)
    float *s_beta = s_alpha + BLOCK_M;                      // (BLOCK_M X 1)
    float *s_q = s_beta + BLOCK_M;                          // (BLOCK_M X d)
    float *s_k = s_q + static_cast<size_t>(BLOCK_M) * d;    // (d X 1)
    float *s_v = s_k + d;                                   // (dv X 1)

    // init acc / l / m
#pragma unroll
    for (size_t i = 0; i < BLOCK_M; ++i) {
        float *s_acc_i = s_acc + (i * dv);
        for (size_t j = tid; j < dv; j += BLOCK_M) {
            s_acc_i[j] = 0.0f;
        }
    }
    if (tid < BLOCK_M) {
        s_l[tid] = 0.0f;
        s_m[tid] = -INFINITY;
    }

    // load q
#pragma unroll
    for (size_t i = 0; i < BLOCK_M; ++i) {
        const size_t qi = q + i;
        float *sq_ptr = s_q + i * d;
        if (qi < seqlen) {
            const T *q_ptr = Q + (qi * nhead + nh) * d;
            for (size_t j = tid; j < d; j += BLOCK_M) {
                sq_ptr[j] = llaisys::utils::nvidia::cast<float>(q_ptr[j]);
            }
        } else {
            for (size_t j = tid; j < d; j += BLOCK_M) {
                sq_ptr[j] = 0.0f;
            }
        }
    }
    __syncthreads();

    // single pass: online softmax
    for (size_t k = 0; k < totlen; ++k) {
        // load k / v
        const T *k_ptr = K + (k * nkvhead + nkvh) * d;
        const T *v_ptr = V + (k * nkvhead + nkvh) * dv;

        for (size_t i = tid; i < d; i += BLOCK_M) {
            s_k[i] = llaisys::utils::nvidia::cast<float>(k_ptr[i]);
        }
        for (size_t i = tid; i < dv; i += BLOCK_M) {
            s_v[i] = llaisys::utils::nvidia::cast<float>(v_ptr[i]);
        }

        __syncthreads();

        if (tid < BLOCK_M) {
            const size_t qi = q + tid;
            const size_t extra = (totlen > seqlen) ? static_cast<size_t>(totlen - seqlen) : 0;
            const size_t limit = min(qi + extra, static_cast<size_t>(totlen) - 1);

            if (qi < seqlen && k <= limit) {
                float dot = 0.0f;
                const float *s_q_i = s_q + (tid * d);

                for (size_t j = 0; j < d; ++j) {
                    dot += s_q_i[j] * s_k[j];
                }

                const float s = dot * scale;
                const float m_old = s_m[tid];
                const float m_new = fmaxf(m_old, s);
                const float alpha = expf(m_old - m_new);
                const float beta = expf(s - m_new);

                s_m[tid] = m_new;
                s_l[tid] = s_l[tid] * alpha + beta;
                s_alpha[tid] = alpha;
                s_beta[tid] = beta;
            } else {
                s_alpha[tid] = 1.0f;
                s_beta[tid] = 0.0f;
            }
        }

        __syncthreads();

#pragma unroll
        for (size_t i = 0; i < BLOCK_M; ++i) {
            const size_t qi = q + i;
            float *s_acc_i = s_acc + (i * dv);

            const size_t extra = (totlen > seqlen) ? static_cast<size_t>(totlen - seqlen) : 0;
            const size_t limit = min(qi + extra, static_cast<size_t>(totlen) - 1);
            if (qi >= seqlen || k > limit) {
                continue;
            }

            const float alpha = s_alpha[i];
            const float beta = s_beta[i];

            for (size_t j = tid; j < dv; j += BLOCK_M) {
                s_acc_i[j] = s_acc_i[j] * alpha + beta * s_v[j];
            }
        }

        __syncthreads();
    }

    // write out
#pragma unroll
    for (size_t i = 0; i < BLOCK_M; ++i) {
        const size_t qi = q + i;
        float *s_acc_i = s_acc + (i * dv);

        if (qi >= seqlen) {
            continue;
        }
        const float den = s_l[i];

        for (size_t j = tid; j < dv; j += BLOCK_M) {
            const float out = (den > 0.0f) ? (s_acc_i[j] / den) : 0.0f;
            const size_t idx = (qi * nhead + nh) * dv + j;

            attn[idx] = llaisys::utils::nvidia::cast<T>(out);
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

    dim3 blockDim(BLOCK_M);
    dim3 gridDim((seqlen + BLOCK_M - 1) / BLOCK_M, nhead);

    const size_t smem_bytes = sizeof(float) * (BLOCK_M * (d + dv + 4) + d + dv);

    flash_attention_kernel<<<gridDim, blockDim, smem_bytes>>>(d_attn, d_q, d_k, d_v, scale, seqlen, nhead, dv, d, total_len, nkvhead);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
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