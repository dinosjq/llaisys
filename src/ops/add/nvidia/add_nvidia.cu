#include "add_nvidia.cuh"

#include "utils/check.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {

template <typename T>
__global__ void add_kernel(T *__restrict__ c, const T *__restrict__ a, const T *__restrict__ b, size_t numel) {
    size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    size_t idx = tid << 2;
    if (idx + 3 < numel) {
        float4 flo4_a = llaisys::utils::nvidia::load_4d(a + idx);
        float4 flo4_b = llaisys::utils::nvidia::load_4d(b + idx);
        float4 flo4_c = llaisys::utils::nvidia::add(flo4_a, flo4_b);
        llaisys::utils::nvidia::save_4d(c + idx, flo4_c);
    } else {
        for (size_t i = idx; i < numel; ++i) {
            float val = llaisys::utils::nvidia::cast<float>(a[i]) + llaisys::utils::nvidia::cast<float>(b[i]);
            c[i] = llaisys::utils::nvidia::cast<T>(val);
        }
    }
}

template <typename T>
void add_launch(std::byte *c, const std::byte *a, const std::byte *b, size_t numel) {
    dim3 blockDim(256);
    dim3 gridDim((((numel + 3) >> 2) + 255) >> 8);

    // 这里传递的数据已经就是在device端的了
    auto *d_c = reinterpret_cast<T *>(c);
    const auto *d_a = reinterpret_cast<const T *>(a);
    const auto *d_b = reinterpret_cast<const T *>(b);

    add_kernel<<<gridDim, blockDim>>>(d_c, d_a, d_b, numel);

    CUDA_CHECK(cudaGetLastError());
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
