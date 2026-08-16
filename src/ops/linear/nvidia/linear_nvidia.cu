#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "gemv_nvidia.cuh"
#include "utils/nvidia_utils.cuh"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t err = (call);                               \
        if (err != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLAS error at "                        \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace {

bool force_cublas() {
    const char *env = std::getenv("LLAISYS_LINEAR_BACKEND");
    return env != nullptr && std::strcmp(env, "cublas") == 0;
}

bool use_gemv(llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    return !force_cublas()
           && llaisys::ops::nvidia::gemv_supported(dtype, m, n, k);
}

template <typename T>
void gemm_launch(T *out, const T *in, const T *weight,
                 size_t m, size_t n, size_t k) {
    cudaDataType_t data_type;
    if constexpr (std::is_same_v<T, float>) {
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        data_type = CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        data_type = CUDA_R_16BF;
    } else {
        throw std::runtime_error("Unsupported data type for cuBLAS GEMM");
    }

    static cublasHandle_t handle = nullptr;
    static cublasMath_t current_math_mode = CUBLAS_DEFAULT_MATH;
    if (handle == nullptr) {
        CUBLAS_CHECK(cublasCreate(&handle));
    }
    const cublasMath_t wanted_math_mode = data_type == CUDA_R_32F
                                              ? CUBLAS_DEFAULT_MATH
                                              : CUBLAS_TENSOR_OP_MATH;
    if (wanted_math_mode != current_math_mode) {
        CUBLAS_CHECK(cublasSetMathMode(handle, wanted_math_mode));
        current_math_mode = wanted_math_mode;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        static_cast<int>(n),
        static_cast<int>(m),
        static_cast<int>(k),
        &alpha,
        reinterpret_cast<const void *>(weight), data_type, static_cast<int>(k),
        reinterpret_cast<const void *>(in), data_type, static_cast<int>(k),
        &beta,
        reinterpret_cast<void *>(out), data_type, static_cast<int>(n),
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT));
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, size_t m, size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    const size_t total = m * n;
    for (size_t idx = tid; idx < total; idx += stride) {
        const size_t col = idx % n;
        const float value = llaisys::utils::nvidia::cast<float>(out[idx])
                            + llaisys::utils::nvidia::cast<float>(bias[col]);
        out[idx] = llaisys::utils::nvidia::cast<T>(value);
    }
}

template <typename T>
void linear_launch(std::byte *d_out, const std::byte *d_in,
                   const std::byte *d_weight, const std::byte *d_bias,
                   llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    if (use_gemv(dtype, m, n, k)) {
        llaisys::ops::nvidia::gemv(d_out, d_in, d_weight, d_bias, dtype, m, n, k);
        return;
    }

    auto *out = reinterpret_cast<T *>(d_out);
    const auto *in = reinterpret_cast<const T *>(d_in);
    const auto *weight = reinterpret_cast<const T *>(d_weight);
    gemm_launch(out, in, weight, m, n, k);

    if (d_bias != nullptr) {
        const auto *bias = reinterpret_cast<const T *>(d_bias);
        constexpr unsigned threads = 256;
        const unsigned blocks = static_cast<unsigned>((m * n + threads - 1) / threads);
        add_bias_kernel<T><<<blocks, threads>>>(out, bias, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
}

} // namespace

namespace llaisys::ops::nvidia {

void linear(std::byte *out, const std::byte *in, const std::byte *weight,
            const std::byte *bias, llaisysDataType_t dtype,
            size_t m, size_t n, size_t k) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_launch<float>(out, in, weight, bias, dtype, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_launch<llaisys::bf16_t>(out, in, weight, bias, dtype, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_launch<llaisys::fp16_t>(out, in, weight, bias, dtype, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
