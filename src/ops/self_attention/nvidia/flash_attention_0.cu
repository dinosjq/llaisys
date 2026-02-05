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

/**
 * flash attention:
 * 实现思路: 固定 (batch, head, q) 为一个 block, 两次遍历 K/V 计算 max 与 sum / acc,
 * 然后输出 softmax(Q @ K^T) @ V.
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
    const int q = blockIdx.x;
    const int nh = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;
    const int nkvh = nh / (nhead / nkvhead);
    const float scale = 1.0f / sqrtf(static_cast<float>(d));

    extern __shared__ float smem[];
    float *s_acc = smem;      // size d
    float *s_sum = s_acc + d; // size 1
    float *s_max = s_sum + 1; // size 1
    float *s_red = s_max + 1; // size blockDim.x

    for (int i = tid; i < d; i += blockDim.x) {
        s_acc[i] = 0.0f;
    }
    if (tid == 0) {
        *s_sum = 0.0f;
        *s_max = -INFINITY;
    }
    __syncthreads();

    const T *q_ptr = Q + ((b * seqlen + q) * nhead + nh) * d;

    // pass 1: max
    float local_max = -INFINITY;
    for (int k = tid; k < totlen; k += blockDim.x) {
        if (is_causal && k > q) {
            continue;
        }
        const T *k_ptr = K + ((b * totlen + k) * nkvhead + nkvh) * d;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += to_float(q_ptr[i]) * to_float(k_ptr[i]);
        }
        const float s = dot * scale;
        local_max = fmaxf(local_max, s);
    }

    s_red[tid] = local_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_red[tid] = fmaxf(s_red[tid], s_red[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        *s_max = s_red[0];
    }
    __syncthreads();

    // pass 2: sum exp + acc
    const float maxv = *s_max;
    for (int k = tid; k < totlen; k += blockDim.x) {
        if (is_causal && k > q) {
            continue;
        }
        const T *k_ptr = K + ((b * totlen + k) * nkvhead + nkvh) * d;
        const T *v_ptr = V + ((b * totlen + k) * nkvhead + nkvh) * d;
        float dot = 0.0f;
        for (int i = 0; i < d; ++i) {
            dot += to_float(q_ptr[i]) * to_float(k_ptr[i]);
        }
        const float s = dot * scale;
        const float p = expf(s - maxv);
        atomicAdd(s_sum, p);
        for (int i = 0; i < d; ++i) {
            atomicAdd(&s_acc[i], p * to_float(v_ptr[i]));
        }
    }
    __syncthreads();

    const float fenmu = *s_sum;
    for (int i = tid; i < d; i += blockDim.x) {
        const float out = (fenmu > 0.0f) ? (s_acc[i] / fenmu) : 0.0f;
        const int idxA = ((b * seqlen + q) * nhead + nh) * d + i;
        A[idxA] = from_float<T>(out);
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
    dim3 gridDim(seqlen, nhead, batch_size);

    const size_t smem_bytes = sizeof(float) * (static_cast<size_t>(d) + 2 + static_cast<size_t>(blockDim.x));

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
