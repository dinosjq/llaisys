#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t err = (call);                               \
        if (err != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLASLt error at "                      \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace {

template <typename T>
void gemm_launch(T *out, const T *in, const T *weight,
                 const size_t M, const size_t N, const size_t K) {
    cudaDataType_t data_type;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if constexpr (std::is_same_v<T, float>) {
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        data_type = CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        data_type = CUDA_R_16BF;
    } else {
        throw std::runtime_error("Unsupported data type for cuBLAS GEMM");
    }

    // Single static handle, created once and reused across all calls.
    // Previously cached per-stream; now all calls use nullptr so a single handle suffices.
    static cublasHandle_t handle = nullptr;
    static bool handle_inited = false;
    if (!handle_inited) {
        CUBLAS_CHECK(cublasCreate(&handle));
        if (data_type == CUDA_R_16F || data_type == CUDA_R_16BF) {
            CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
        }
        handle_inited = true;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,         // weight 做转置
        CUBLAS_OP_N,         // in 不转置
        static_cast<int>(N), // m
        static_cast<int>(M), // n
        static_cast<int>(K), // k
        &alpha,
        reinterpret_cast<const void *>(weight), data_type, static_cast<int>(K), // lda for weight
        reinterpret_cast<const void *>(in), data_type, static_cast<int>(K),     // ldb for in
        &beta,
        reinterpret_cast<void *>(out), data_type, static_cast<int>(N), // ldc
        compute_type,
        CUBLAS_GEMM_DEFAULT));
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, const size_t M, const size_t N) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
    const size_t total = M * N;

    for (size_t idx = tid; idx < total; idx += stride) {
        const size_t col = idx % N;
        const float v = llaisys::utils::nvidia::cast<float>(out[idx]) + llaisys::utils::nvidia::cast<float>(bias[col]);
        out[idx] = llaisys::utils::nvidia::cast<T>(v);
    }
}


template <typename T>
void linear_launch(std::byte *d_out, const std::byte *d_in, const std::byte *d_weight, const std::byte *d_bias,
                   const size_t m, const size_t n, const size_t k) {
    T *out = reinterpret_cast<T *>(d_out);
    const T *in = reinterpret_cast<const T *>(d_in);
    const T *weight = reinterpret_cast<const T *>(d_weight);
    const T *bias = reinterpret_cast<const T *>(d_bias);

    /** END: cublas调用gemm + 自定义转置加和核函数 */
    gemm_launch(out, in, weight, m, n, k);

    // add bias
    if (d_bias != nullptr) {
        dim3 blockDim(256);
        dim3 gridDim((m * n + blockDim.x - 1) / blockDim.x);
        add_bias_kernel<T><<<gridDim, blockDim>>>(out, bias, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type,
            const size_t m, const size_t n, const size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_launch<float>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_launch<llaisys::bf16_t>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_launch<llaisys::fp16_t>(out, in, weight, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
