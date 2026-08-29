#include "linear_w8a8_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <stdexcept>

#define W8_CUBLAS_CHECK(call)                                      \
    do {                                                           \
        cublasStatus_t err = (call);                               \
        if (err != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLAS error at "                        \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace {

cublasHandle_t w8_handle() {
    static cublasHandle_t handle = nullptr;
    if (handle == nullptr) {
        W8_CUBLAS_CHECK(cublasCreate(&handle));
    }
    return handle;
}

// int32 GEMM 累积工作区 (算子内部静态，按需增长)
int *acc_workspace(size_t bytes) {
    static std::mutex mu;
    static int *ptr = nullptr;
    static size_t capacity = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (bytes > capacity) {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            capacity = 0;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), bytes));
        capacity = bytes;
    }
    return ptr;
}

// epilogue：out[m,n] = acc_i32 * a_scale[row] * w_scale[col] (+ bias[col])，cast 回 T
template <typename T>
__global__ void epilogue_kernel(T *__restrict__ out, 
                                const int *__restrict__ acc,
                                const float *__restrict__ a_scale, 
                                const float *__restrict__ w_scale,
                                const T *__restrict__ bias, 
                                size_t m, 
                                size_t n) 
{
    const size_t total = m * n;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < total; i += stride) {
        const size_t row = i / n;
        const size_t col = i % n;
        float v = static_cast<float>(acc[i]) * a_scale[row] * w_scale[col];
        if (bias != nullptr) {
            v += llaisys::utils::nvidia::cast<float>(bias[col]);
        }
        out[i] = llaisys::utils::nvidia::cast<T>(v);
    }
}

// dp4a INT8 GEMV (decode 小 M，替代 cuBLAS M 很小时的 launch 开销)
// out[m,n] = a_scale[m] * w_scale[n] * Σ_k a_i8[m,k] * w_i8[n,k] (+ bias[n])。
// 每 warp 负责一列 (grid-stride): lane 按 4 字节块做 __dp4a；m(≤8) 行复用同一份 weight 寄存器累加。
template <typename T>
__global__ void gemv_w8a8_kernel(T *__restrict__ out, 
                                 const int8_t *__restrict__ a_i8,
                                 const float *__restrict__ a_scale, 
                                 const int8_t *__restrict__ w_i8,
                                 const float *__restrict__ w_scale, 
                                 const T *__restrict__ bias,
                                 size_t m, 
                                 size_t n, 
                                 size_t k) 
{
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;

    for (size_t col = static_cast<size_t>(blockIdx.x) * nwarps + warp; col < n; col += static_cast<size_t>(gridDim.x) * nwarps) {
        const int8_t *w_col = w_i8 + col * k;
        int32_t acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (size_t kk = static_cast<size_t>(lane) * 4; kk < k; kk += 128) {
            const int32_t w4 = *reinterpret_cast<const int32_t *>(w_col + kk);
#pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (static_cast<size_t>(r) < m) {
                    const int32_t a4 = *reinterpret_cast<const int32_t *>(a_i8 + static_cast<size_t>(r) * k + kk);
                    acc[r] = __dp4a(a4, w4, acc[r]); // 3 参形式：a·b + c
                }
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            if (static_cast<size_t>(r) < m) {
                int32_t v = acc[r];
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) {
                    v += __shfl_xor_sync(0xffffffffu, v, off);
                }
                if (lane == 0) {
                    float f = static_cast<float>(v) * a_scale[r] * w_scale[col];
                    if (bias != nullptr) {
                        f += llaisys::utils::nvidia::cast<float>(bias[col]);
                    }
                    out[r * n + col] = llaisys::utils::nvidia::cast<T>(f);
                }
            }
        }
    }
}

// dp4a INT8 GEMV v2: smem 预加载激活 (消除每 warp 重复读 act) + int4 向量化权重读 + m 模板化
// 每 warp 一列 (grid-stride): lane 按 16 字节块一次做 4 个 __dp4a
// 要求 k % 16 == 0 (Hadamard 路径 K 恒为 128 倍数，满足)
template <typename T, int M>
__global__ void gemv_w8a8_v2_kernel(T *__restrict__ out, const int8_t *__restrict__ a_i8,
                                    const float *__restrict__ a_scale, const int8_t *__restrict__ w_i8,
                                    const float *__restrict__ w_scale, const T *__restrict__ bias,
                                    size_t n, size_t k) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    const int k4 = static_cast<int>(k >> 2);

    // smem 预加载 act：M*k 个 int8 (打包成 int32) + M 个 a_scale
    extern __shared__ int32_t s_a[];
    for (int i = threadIdx.x; i < M * k4; i += blockDim.x) {
        s_a[i] = reinterpret_cast<const int32_t *>(a_i8)[i];
    }
    float *s_a_scale = reinterpret_cast<float *>(s_a + M * k4);
    if (threadIdx.x < M) {
        s_a_scale[threadIdx.x] = a_scale[threadIdx.x];
    }
    __syncthreads();

    for (size_t col = static_cast<size_t>(blockIdx.x) * nwarps + warp; col < n;
         col += static_cast<size_t>(gridDim.x) * nwarps) {
        const int8_t *w_col = w_i8 + col * k;
        int32_t acc[M] = {0};
        // int4 向量化：每 lane 一次读 16 字节 (16 个 int8), 做 4 个 __dp4a
        for (size_t kk = static_cast<size_t>(lane) * 16; kk < k; kk += static_cast<size_t>(32) * 16) {
            const int4 wv = *reinterpret_cast<const int4 *>(w_col + kk);
            const int a4 = static_cast<int>(kk >> 2);
#pragma unroll
            for (int r = 0; r < M; ++r) {
                const int32_t *ar = s_a + r * k4;
                acc[r] = __dp4a(ar[a4 + 0], wv.x, acc[r]);
                acc[r] = __dp4a(ar[a4 + 1], wv.y, acc[r]);
                acc[r] = __dp4a(ar[a4 + 2], wv.z, acc[r]);
                acc[r] = __dp4a(ar[a4 + 3], wv.w, acc[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < M; ++r) {
            int32_t v = acc[r];
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v += __shfl_xor_sync(0xffffffffu, v, off);
            }
            if (lane == 0) {
                float f = static_cast<float>(v) * s_a_scale[r] * w_scale[col];
                if (bias != nullptr) {
                    f += llaisys::utils::nvidia::cast<float>(bias[col]);
                }
                out[r * n + col] = llaisys::utils::nvidia::cast<T>(f);
            }
        }
    }
}

template <typename T>
void launch_v2(std::byte *out, const std::byte *in_q, const std::byte *a_scale, const std::byte *weight_i8,
               const std::byte *w_scale, const std::byte *bias, size_t m, size_t n, size_t k) {
    const int threads = 512; // 16 warp：更多列并行 + smem 复用 act，降低 act 冗余读取
    const int nwarps = threads >> 5;
    const int blocks = static_cast<int>((n + nwarps - 1) / nwarps);
    const size_t smem = m * (k >> 2) * 4 + m * 4; // M*k int8 + M float scale
    auto *o = reinterpret_cast<T *>(out);
    auto *a = reinterpret_cast<const int8_t *>(in_q);
    auto *as = reinterpret_cast<const float *>(a_scale);
    auto *w = reinterpret_cast<const int8_t *>(weight_i8);
    auto *ws = reinterpret_cast<const float *>(w_scale);
    auto *b = reinterpret_cast<const T *>(bias);
    switch (m) {
#define LLAISYS_LAUNCH_M(v) \
    case v: \
        gemv_w8a8_v2_kernel<T, v><<<blocks, threads, smem>>>(o, a, as, w, ws, b, n, k); \
        break;
        LLAISYS_LAUNCH_M(1)
        LLAISYS_LAUNCH_M(2)
        LLAISYS_LAUNCH_M(3)
        LLAISYS_LAUNCH_M(4)
        LLAISYS_LAUNCH_M(5)
        LLAISYS_LAUNCH_M(6)
        LLAISYS_LAUNCH_M(7)
        LLAISYS_LAUNCH_M(8)
#undef LLAISYS_LAUNCH_M
    default:
        return;
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch(std::byte *out, const std::byte *in_q, const std::byte *a_scale, const std::byte *weight_i8,
            const std::byte *w_scale, const std::byte *bias, size_t m, size_t n, size_t k) {
    // decode 小 M (≤8) 且 K 是 4 的倍数: 手写 dp4a GEMV，避免 cuBLAS 小 M launch 开销
    // int32 累加与 cuBLAS CUBLAS_COMPUTE_32I 逐位一致 (整数加法无舍入) epilogue 同算式
    // K%16==0 时走 v2 (smem 复用 act + int4 向量化) 否则走原版 (int32 标量读)
    if (m <= 8 && (k % 16) == 0 && m * (k >> 2) * 4 + m * 4 <= 48 * 1024) {
        return launch_v2<T>(out, in_q, a_scale, weight_i8, w_scale, bias, m, n, k);
    }
    if (m <= 8 && (k % 4) == 0) {
        const int threads = 256;
        const int nwarps = threads >> 5;
        const int blocks = static_cast<int>((n + nwarps - 1) / nwarps);
        gemv_w8a8_kernel<T><<<blocks, threads>>>(
            reinterpret_cast<T *>(out), reinterpret_cast<const int8_t *>(in_q),
            reinterpret_cast<const float *>(a_scale), reinterpret_cast<const int8_t *>(weight_i8),
            reinterpret_cast<const float *>(w_scale), reinterpret_cast<const T *>(bias), m, n, k);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    // 1. int32 累积工作区
    int *acc = acc_workspace(m * n * sizeof(int));

    // 2. cuBLAS INT8 GEMM：acc[m,n] = in_q[m,k] × weight_i8[n,k]^T (int32 累积)
    const int32_t alpha = 1;
    const int32_t beta = 0;
    W8_CUBLAS_CHECK(cublasGemmEx(
        w8_handle(), CUBLAS_OP_T, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(m), static_cast<int>(k),
        &alpha, weight_i8, CUDA_R_8I, static_cast<int>(k), in_q, CUDA_R_8I, static_cast<int>(k), &beta,
        acc, CUDA_R_32I, static_cast<int>(n), CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT));

    // 3. epilogue：int32 → FP (乘 a_scale × w_scale + bias)
    const size_t total = m * n;
    const int blocks = static_cast<int>((total + 255) / 256);
    epilogue_kernel<T><<<blocks, 256>>>(reinterpret_cast<T *>(out), acc, reinterpret_cast<const float *>(a_scale),
                                        reinterpret_cast<const float *>(w_scale),
                                        reinterpret_cast<const T *>(bias), m, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void linear_w8a8(std::byte *out, const std::byte *in_q, const std::byte *a_scale, const std::byte *weight_i8,
                 const std::byte *w_scale, const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t n,
                 size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, in_q, a_scale, weight_i8, w_scale, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out, in_q, a_scale, weight_i8, w_scale, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out, in_q, a_scale, weight_i8, w_scale, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
