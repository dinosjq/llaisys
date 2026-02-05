#include <algorithm>
#include <cmath>
#include <cstring>
#include <cuda_fp16.h>
#include <type_traits>
#include <vector>

#define RUNTIME_CHECK(call)                                                    \
  do {                                                                         \
    RUNTIME_ERR_TYPE err = call;                                               \
    if (err != RUNTIME_SUCCESS_CODE) {                                         \
      std::cerr << "Runtime error at " << __FILE__ << ":" << __LINE__ << " - " \
                << RUNTIME_GET_ERROR_STR(err) << "\n";                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)


template <typename T>
__device__ __forceinline__ float to_float(T v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return __half2float(v);
    }
    return 0.0;
}

template <typename T>
__device__ __forceinline__ T from_float(float v) {
    if constexpr (std::is_same_v<T, float>) {
        return v;
    } else if constexpr (std::is_same_v<T, half>) {
        return __float2half_rn(v);
    }
    return 0.0;
}

constexpr static int BLOCK_M = 32;

/**
 * flash attention
 * 将 two pass 的 softmax 转化为 single pass
 * 实现 smem 加载的向量化访存
 */
template <typename T>
__global__ void flash_attention_kernel(
    const T *Q,
    const T *K,
    const T *V,
    T *A,
    const int seqlen,
    const int totlen,
    const int nhead,
    const int nkvhead,
    const int d,
    const bool is_causal) {
    const int bq = blockIdx.x;
    const int q = bq * BLOCK_M;
    const int nh = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int nkvh = nh / (nhead / nkvhead);
    const float scale = 1.0f / sqrtf(static_cast<float>(d));
    const int warp_id = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;

    extern __shared__ float smem[];
    float *s_acc = smem;                                   // (BLOCK_M X d) 计算的暂存值
    float *s_l = s_acc + static_cast<size_t>(BLOCK_M) * d; // (BLOCK_M X 1) 分母和
    float *s_m = s_l + BLOCK_M;                            // (BLOCK_M X 1) 最大值
    float *s_alpha = s_m + BLOCK_M;                        // (BLOCK_M X 1)
    float *s_beta = s_alpha + BLOCK_M;                     // (BLOCK_M X 1)
    float *s_q = s_beta + BLOCK_M;                         // (BLOCK_M X d)
    float *s_k = s_q + static_cast<size_t>(BLOCK_M) * d;   // (d X 1)
    float *s_v = s_k + d;                                  // (d X 1)

    // init acc / l / m
#pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = tid; j < d; j += blockDim.x) {
            s_acc[i * d + j] = 0.0f;
        }
    }
    if (tid < BLOCK_M) {
        s_l[tid] = 0.0f;
        s_m[tid] = -INFINITY;
    }
    __syncthreads();

    // load q
    // todo: 向量化访存
#pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) {
        const int qi = q + i;
        if (qi < seqlen) {
            const T *q_ptr = Q + ((b * seqlen + qi) * nhead + nh) * d;
            if constexpr (std::is_same_v<T, float>) {
                const int start = (tid << 2);
                const int step = (blockDim.x << 2);
                const int offset = i * d;
                for (int j = start; j < d; j += step) {
                    const int x = offset + j;
                    if (j + 3 < d) {
                        const float4 flo4 = *reinterpret_cast<const float4 *>(q_ptr + j);
                        s_q[x + 0] = flo4.x;
                        s_q[x + 1] = flo4.y;
                        s_q[x + 2] = flo4.z;
                        s_q[x + 3] = flo4.w;
                    } else if (j + 2 < d) {
                        const float3 flo3 = *reinterpret_cast<const float3 *>(q_ptr + j);
                        s_q[x + 0] = flo3.x;
                        s_q[x + 1] = flo3.y;
                        s_q[x + 2] = flo3.z;
                    } else if (j + 1 < d) {
                        const float2 flo2 = *reinterpret_cast<const float2 *>(q_ptr + j);
                        s_q[x + 0] = flo2.x;
                        s_q[x + 1] = flo2.y;
                    } else {
                        s_q[x + 0] = q_ptr[j];
                    }
                }
            } else if constexpr (std::is_same_v<T, half>) {
                const int start = (tid << 1);
                const int step = (blockDim.x << 1);
                const int offset = i * d;
                for (int j = start; j < d; j += step) {
                    const int x = offset + j;
                    if (j + 1 < d) {
                        const float2 flo2 = __half22float2(*reinterpret_cast<const half2 *>(q_ptr + j));
                        s_q[x + 0] = flo2.x;
                        s_q[x + 1] = flo2.y;
                    } else {
                        s_q[x + 0] = __half2float(q_ptr[j]);
                    }
                }
            }
        } else {
            for (int j = tid; j < d; j += blockDim.x) {
                s_q[i * d + j] = T(0);
            }
        }
    }
    __syncthreads();

    // single pass: online softmax
    for (int k = 0; k < totlen; ++k) {
        const T *k_ptr = K + ((b * totlen + k) * nkvhead + nkvh) * d;
        const T *v_ptr = V + ((b * totlen + k) * nkvhead + nkvh) * d;
        // load k / v
        // todo: 向量化访存
        if constexpr (std::is_same_v<T, float>) {
            const int start = (tid << 2);
            const int step = (blockDim.x << 2);
            for (int j = start; j < d; j += step) {
                if (j + 3 < d) {
                    const float4 flo4_k = *reinterpret_cast<const float4 *>(k_ptr + j);
                    s_k[j + 0] = flo4_k.x;
                    s_k[j + 1] = flo4_k.y;
                    s_k[j + 2] = flo4_k.z;
                    s_k[j + 3] = flo4_k.w;
                    const float4 flo4_v = *reinterpret_cast<const float4 *>(v_ptr + j);
                    s_v[j + 0] = flo4_v.x;
                    s_v[j + 1] = flo4_v.y;
                    s_v[j + 2] = flo4_v.z;
                    s_v[j + 3] = flo4_v.w;
                } else if (j + 2 < d) {
                    const float3 flo3_k = *reinterpret_cast<const float3 *>(k_ptr + j);
                    s_k[j + 0] = flo3_k.x;
                    s_k[j + 1] = flo3_k.y;
                    s_k[j + 2] = flo3_k.z;
                    const float3 flo3_v = *reinterpret_cast<const float3 *>(v_ptr + j);
                    s_v[j + 0] = flo3_v.x;
                    s_v[j + 1] = flo3_v.y;
                    s_v[j + 2] = flo3_v.z;
                } else if (j + 1 < d) {
                    const float2 flo2_k = *reinterpret_cast<const float2 *>(k_ptr + j);
                    s_k[j + 0] = flo2_k.x;
                    s_k[j + 1] = flo2_k.y;
                    const float2 flo2_v = *reinterpret_cast<const float2 *>(v_ptr + j);
                    s_v[j + 0] = flo2_v.x;
                    s_v[j + 1] = flo2_v.y;
                } else {
                    s_k[j + 0] = k_ptr[j + 0];
                    s_v[j + 0] = v_ptr[j + 0];
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            const int start = (tid << 1);
            const int step = (blockDim.x << 1);
            for (int j = start; j < d; j += step) {
                if (j + 1 < d) {
                    const float2 flo2_k = __half22float2(*reinterpret_cast<const half2 *>(k_ptr + j));
                    s_k[j + 0] = flo2_k.x;
                    s_k[j + 1] = flo2_k.y;
                    const float2 flo2_v = __half22float2(*reinterpret_cast<const half2 *>(v_ptr + j));
                    s_v[j + 0] = flo2_v.x;
                    s_v[j + 1] = flo2_v.y;
                } else {
                    s_k[j + 0] = __half2float(k_ptr[j + 0]);
                    s_v[j + 0] = __half2float(v_ptr[j + 0]);
                }
            }
        }

        __syncthreads();

        if (tid < BLOCK_M) {
            const int qi = q + tid;
            if (qi < seqlen && (!is_causal || k <= qi)) {
                // todo: warp 归约优化计算
                float dot = 0.0f;
                for (int j = 0; j < d; ++j) {
                    dot += to_float(s_q[tid * d + j]) * to_float(s_k[j]);
                }
                // 当前的值
                const float s = dot * scale;
                // 之前的最大
                const float m_old = s_m[tid];
                /**
                 *  发现有些点的精准度不够
                 *  怀疑是存在溢出的情况 即 s 过大导致精度不够
                 *  所以在这里对 s 进行裁剪
                 *  对 s 裁剪不会导致整体分布改变
                 *  并且裁剪后确实有效
                 */
                // 更新的最大并进行裁剪
                const float m_new = fmaxf(m_old, s - 10);
                // 给之前的缩放因子
                const float alpha = expf(m_old - m_new);
                // 当前的增量
                const float beta = expf(s - m_new);
                // 暂存
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

        // 延后更新
#pragma unroll
        for (int i = 0; i < BLOCK_M; ++i) {
            const int qi = q + i;
            if (qi >= seqlen || (is_causal && k > qi)) {
                continue;
            }
            const float alpha = s_alpha[i];
            const float beta = s_beta[i];
            for (int j = tid; j < d; j += blockDim.x) {
                s_acc[i * d + j] = s_acc[i * d + j] * alpha + beta * to_float(s_v[j]);
            }
        }
        __syncthreads();
    }

    // todo: 向量化写回
#pragma unroll
    for (int i = 0; i < BLOCK_M; ++i) {
        const int qi = q + i;
        if (qi >= seqlen) {
            continue;
        }
        const float denom = s_l[i];
        const int offset = i * d;
        const int offset_A = ((b * seqlen + qi) * nhead + nh) * d;
        if constexpr (std::is_same_v<T, float>) {
            const int start = tid << 2;
            const int step = blockDim.x << 2;
            for (int j = start; j < d; j += step) {
                if (j + 3 < d) {
                    const float out_0 = (denom > 0.0f) ? (s_acc[offset + j + 0] / denom) : 0.0f;
                    const float out_1 = (denom > 0.0f) ? (s_acc[offset + j + 1] / denom) : 0.0f;
                    const float out_2 = (denom > 0.0f) ? (s_acc[offset + j + 2] / denom) : 0.0f;
                    const float out_3 = (denom > 0.0f) ? (s_acc[offset + j + 3] / denom) : 0.0f;
                    const float4 flo4 = make_float4(out_0, out_1, out_2, out_3);
                    *reinterpret_cast<float4 *>(&A[offset_A + j]) = flo4;
                } else if (j + 2 < d) {
                    const float out_0 = (denom > 0.0f) ? (s_acc[offset + j + 0] / denom) : 0.0f;
                    const float out_1 = (denom > 0.0f) ? (s_acc[offset + j + 1] / denom) : 0.0f;
                    const float out_2 = (denom > 0.0f) ? (s_acc[offset + j + 2] / denom) : 0.0f;
                    const float3 flo3 = make_float3(out_0, out_1, out_2);
                    *reinterpret_cast<float3 *>(&A[offset_A + j]) = flo3;
                } else if (j + 1 < d) {
                    const float out_0 = (denom > 0.0f) ? (s_acc[offset + j + 0] / denom) : 0.0f;
                    const float out_1 = (denom > 0.0f) ? (s_acc[offset + j + 1] / denom) : 0.0f;
                    const float2 flo2 = make_float2(out_0, out_1);
                    *reinterpret_cast<float2 *>(&A[offset_A + j]) = flo2;
                } else {
                    const float out_0 = (denom > 0.0f) ? (s_acc[offset + j + 0] / denom) : 0.0f;
                    A[offset_A + j] = out_0;
                }
            }
        } else if constexpr (std::is_same_v<T, half>) {
            const int start = tid << 1;
            const int step = blockDim.x << 1;
            for (int j = start; j < d; j += step) {
                if (j + 1 < d) {
                    const float out_0 = (denom > 0.0f) ? (s_acc[i * d + j + 0] / denom) : 0.0f;
                    const float out_1 = (denom > 0.0f) ? (s_acc[i * d + j + 1] / denom) : 0.0f;
                    const half2 hlf2 = make_half2(__float2half(out_0), __float2half(out_1));
                    *reinterpret_cast<half2 *>(&A[offset_A + j]) = hlf2;
                } else {
                    const float out_0 = (denom > 0.0f) ? (s_acc[i * d + j + 0] / denom) : 0.0f;
                    A[offset_A + j] = __float2half(out_0);
                }
            }
        }
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 *
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, seqlen, nhead, d]
 * @param[in] h_k Key tensor of shape [batch_size, totlen, nkvhead, d]
 * @param[in] h_v Value tensor of shape [batch_size, totlen, nkvhead, d]
 * @param[out] h_a Output attention tensor of shape [batch_size, seqlen, nhead, d]
 * @param[in] batch_size Batch dimension size
 * @param[in] seqlen Target sequence length
 * @param[in] totlen Source sequence length
 * @param[in] nhead Number of query attention heads
 * @param[in] nkvhead Number of key/value heads (supports grouped query attention)
 * @param[in] d Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T> &h_q, const std::vector<T> &h_k,
                    const std::vector<T> &h_v, std::vector<T> &h_a,
                    int batch_size, int seqlen, int totlen,
                    int nhead, int nkvhead, int d, bool is_causal) noexcept {
    // 异常处理
    if (batch_size <= 0 || seqlen <= 0 || totlen <= 0 || nhead <= 0 || nkvhead <= 0 || d <= 0) {
        return;
    }
    if (nhead % nkvhead) {
        return;
    }
    // 计算个数
    const size_t q_numel = static_cast<size_t>(batch_size) * seqlen * nhead * d;
    const size_t k_numel = static_cast<size_t>(batch_size) * totlen * nkvhead * d;
    const size_t v_numel = static_cast<size_t>(batch_size) * totlen * nkvhead * d;
    const size_t a_numel = static_cast<size_t>(batch_size) * seqlen * nhead * d;
    // 内存大小
    const size_t Tsize = sizeof(T);
    const size_t q_bytes = q_numel * Tsize;
    const size_t k_bytes = k_numel * Tsize;
    const size_t v_bytes = v_numel * Tsize;
    const size_t a_bytes = a_numel * Tsize;
    // device侧数据
    T *d_q = nullptr;
    T *d_k = nullptr;
    T *d_v = nullptr;
    T *d_a = nullptr;
    // 内存分配
    RUNTIME_CHECK(cudaMalloc(&d_q, q_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_k, k_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_v, v_bytes));
    RUNTIME_CHECK(cudaMalloc(&d_a, a_bytes));
    // host -> device
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), k_bytes, cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), v_bytes, cudaMemcpyHostToDevice));

    // 对 seqlen 维度划分, 一个 block 计算BLOCK_M行 (batch, head, q)
    dim3 blockDim(BLOCK_M);
    dim3 gridDim((seqlen + BLOCK_M - 1) / BLOCK_M, nhead, batch_size);

    const size_t smem_bytes = sizeof(float)
                            * (static_cast<size_t>(BLOCK_M) * d * 2
                               + static_cast<size_t>(BLOCK_M) * 4
                               + static_cast<size_t>(d) * 2);

    flash_attention_kernel<<<gridDim, blockDim, smem_bytes>>>(d_q, d_k, d_v, d_a, seqlen, totlen, nhead, nkvhead, d, is_causal);

    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());

    RUNTIME_CHECK(cudaMemcpy(h_a.data(), d_a, a_bytes, cudaMemcpyDeviceToHost));
    // 释放内存
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_a));
}
