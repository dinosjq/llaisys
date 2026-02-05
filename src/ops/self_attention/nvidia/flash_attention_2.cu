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

constexpr static int BLOCK_M = 8;

/**
 * flash attention:
 * 2: 实现seqlen分块 每块8行 实现 k 的共享内存
 * 
 * 由于没法分线程固定 k v 到smem 只能每个线程load到寄存器
 * 由于每个线程都要load k v 的向量 导致带宽使用率降低
 * 反而导致性能缺陷
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

    extern __shared__ float smem[];
    float *s_acc[BLOCK_M]; // size d
    float *s_sum[BLOCK_M]; // size 1
    float *s_max[BLOCK_M]; // size 1
    float *s_red[BLOCK_M]; // size blockDim.x
    float *s_q[BLOCK_M];   // size d
    float s_k[64];         // size d
    float s_v[64];         // size d

    const size_t LAYER_SIZE = static_cast<size_t>(d) + 2 + blockDim.x + d;
    for (int i = 0; i < BLOCK_M; ++i) {
        s_acc[i] = smem + i * LAYER_SIZE;
        s_sum[i] = s_acc[i] + d;
        s_max[i] = s_sum[i] + 1;
        s_red[i] = s_max[i] + 1;
        s_q[i] = s_red[i] + blockDim.x;
    }

    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = tid; j < d; j += blockDim.x) {
            s_acc[i][j] = 0.0f;
        }
    }
    if (tid == 0) {
        for (int i = 0; i < BLOCK_M; ++i) {
            *s_sum[i] = 0.0f;
            *s_max[i] = -INFINITY;
        }
    }
    __syncthreads();

    // load q
    for (int i = 0; i < BLOCK_M; ++i) {
        const int qi = q + i;
        if (qi < seqlen) {
            const T *q_ptr = Q + ((b * seqlen + qi) * nhead + nh) * d;
            for (int j = tid; j < d; j += blockDim.x) {
                s_q[i][j] = q_ptr[j];
            }
        } else {
            for (int j = tid; j < d; j += blockDim.x) {
                s_q[i][j] = T(0);
            }
        }
    }

    for (int i = 0; i < BLOCK_M; ++i) {
        s_red[i][tid] = -INFINITY;
    }
    __syncthreads();

    // pass 1: max
    for (int k = tid; k < totlen; k += blockDim.x) {
        const T *k_ptr = K + ((b * totlen + k) * nkvhead + nkvh) * d;
        // load k
        for (int i = 0; i < d; ++i) {
            s_k[i] = k_ptr[i];
        }

        for (int i = 0; i < BLOCK_M; ++i) {
            const int qi = q + i;
            if (is_causal && k > qi || qi >= seqlen) {
                continue;
            }
            float dot = 0.0f;
            for (int j = 0; j < d; ++j) {
                dot += to_float(s_q[i][j]) * to_float(s_k[j]);
            }
            const float s = dot * scale;
            s_red[i][tid] = fmaxf(s_red[i][tid], s);
        }
    }

    __syncthreads();

    for (int i = 0; i < BLOCK_M; ++i) {
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                s_red[i][tid] = fmaxf(s_red[i][tid], s_red[i][tid + stride]);
            }
            __syncthreads();
        }
        if (tid == 0) {
            *s_max[i] = s_red[i][0];
        }
        __syncthreads();
    }

    // pass 2: sumexp + acc
    for (int k = tid; k < totlen; k += blockDim.x) {
        const T *k_ptr = K + ((b * totlen + k) * nkvhead + nkvh) * d;
        const T *v_ptr = V + ((b * totlen + k) * nkvhead + nkvh) * d;
        // load k / v
        for (int i = 0; i < d; ++i) {
            s_k[i] = k_ptr[i];
            s_v[i] = v_ptr[i];
        }

        float sum[BLOCK_M] = {0.0f};
        for (int i = 0; i < BLOCK_M; ++i) {
            const int qi = q + i;
            if (is_causal && k > qi || qi >= seqlen) {
                continue;
            }
            const float maxv = *s_max[i];
            float dot = 0.0f;
            for (int j = 0; j < d; ++j) {
                dot += to_float(s_q[i][j]) * to_float(s_k[j]);
            }
            const float s = dot * scale;
            const float p = expf(s - maxv);
            sum[i] += p;
            for (int j = 0; j < d; ++j) {
                atomicAdd(&s_acc[i][j], p * to_float(s_v[j]));
            }
        }
        for (int i = 0; i < BLOCK_M; ++i) {
            atomicAdd(s_sum[i], sum[i]);
        }
        __syncthreads();
    }

    for (int i = 0; i < BLOCK_M; ++i) {
        const int qi = q + i;
        if (qi >= seqlen) {
            continue;
        }
        const float fenmu = *s_sum[i];
        for (int j = tid; j < d; j += blockDim.x) {
            const float out = (fenmu > 0.0f) ? (s_acc[i][j] / fenmu) : 0.0f;
            const int idxA = ((b * seqlen + qi) * nhead + nh) * d + j;
            A[idxA] = from_float<T>(out);
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

    // 对 seqlen 维度划分, 一个 block 计算一行 (batch, head, q)
    dim3 blockDim(256);
    dim3 gridDim((seqlen + BLOCK_M - 1) / BLOCK_M, nhead, batch_size);

    const size_t smem_bytes = sizeof(float)
                            * ((static_cast<size_t>(d) + 2 + blockDim.x + d) * BLOCK_M);

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