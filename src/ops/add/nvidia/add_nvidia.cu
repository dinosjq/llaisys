#include "add_nvidia.cuh"

#include "utils/check.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
template <typename T>
__device__ __forceinline__ T add_value(T a, T b) {
    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
        float sum = llaisys::utils::nvidia::cast<float>(a) + llaisys::utils::nvidia::cast<float>(b);
        return llaisys::utils::nvidia::cast<T>(sum);
    } else {
        return a + b;
    }
}

template <typename T>
__global__ void add_kernel(T *c, const T *a, const T *b, size_t numel) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx < numel) {
        c[idx] = add_value(a[idx], b[idx]);
    }
}

template <typename T>
void add_launch(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    dim3 blockDim(256);
    dim3 gridDim((numel + blockDim.x - 1) / blockDim.x);
    // 这里传递的数据已经就是在device端的了
    auto *d_c = reinterpret_cast<T *>(c);
    const auto *d_a = reinterpret_cast<const T *>(a);
    const auto *d_b = reinterpret_cast<const T *>(b);
    add_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, numel);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}
} // namespace

namespace llaisys::ops::nvidia {
void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return add_launch<float>(c, a, b, numel);
    case LLAISYS_DTYPE_BF16:
        return add_launch<llaisys::bf16_t>(c, a, b, numel);
    case LLAISYS_DTYPE_F16:
        return add_launch<llaisys::fp16_t>(c, a, b, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
