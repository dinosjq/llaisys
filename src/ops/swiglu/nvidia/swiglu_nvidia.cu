#include "swiglu_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

template <typename T>
void swiglu_(T *out, const T *gate, const T *up, size_t numel) {
    for (size_t i = 0; i < numel; i++) {
        if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
            const float t = llaisys::utils::cast<float>(up[i]);
            const float s = llaisys::utils::cast<float>(gate[i]);
            out[i] = llaisys::utils::cast<T>(t * s / (1.0f + std::exp(-s)));
        } else {
            out[i] = up[i] * gate[i] / (1 + std::exp(-gate[i]));
        }
    }
}

template <typename T>
__global__ void swiglu_kernel(T *out, const T *gate, const T *up, const size_t numel) {
    const size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < numel){
        const float t = llaisys::utils::nvidia::cast<float>(up[idx]);
        const float s = llaisys::utils::nvidia::cast<float>(gate[idx]);
        out[idx] = llaisys::utils::nvidia::cast<T>(t * s / (1.0f + std::exp(-s)));
    }
}

template <typename T>
void swiglu_launch(std::byte *out, const std::byte *gate, const std::byte *up, const size_t &numel) {
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_gate = reinterpret_cast<const T *>(gate);
    const auto *d_up = reinterpret_cast<const T *>(up);

    dim3 blockDim(256);
    dim3 gridDim((numel + 255) / 256);

    swiglu_kernel<<<gridDim, blockDim>>>(d_out, d_gate, d_up, numel);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, const size_t &numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return swiglu_launch<float>(out, gate, up, numel);
    case LLAISYS_DTYPE_BF16:
        return swiglu_launch<llaisys::bf16_t>(out, gate, up, numel);
    case LLAISYS_DTYPE_F16:
        return swiglu_launch<llaisys::fp16_t>(out, gate, up, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
