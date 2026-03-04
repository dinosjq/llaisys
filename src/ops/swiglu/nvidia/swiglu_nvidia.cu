#include "swiglu_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

__device__ __forceinline__ float4 calc(const float4 &flo4_t, const float4 &flo4_s){
    return make_float4(flo4_t.x * flo4_s.x / (1.0f + expf(-flo4_s.x)),
                       flo4_t.y * flo4_s.y / (1.0f + expf(-flo4_s.y)), 
                       flo4_t.z * flo4_s.z / (1.0f + expf(-flo4_s.z)),
                       flo4_t.w * flo4_s.w / (1.0f + expf(-flo4_s.w)));
}

template <typename T>
__global__ void swiglu_kernel(T *__restrict__ out, const T *__restrict__ gate, const T *__restrict__ up, const size_t numel) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int idx = tid << 2;

    if(idx + 3 < numel){
        const float4 flo4_t = llaisys::utils::nvidia::load_4d(up + idx);
        const float4 flo4_s = llaisys::utils::nvidia::load_4d(gate + idx);
        const float4 flo4_v = calc(flo4_t, flo4_s);
        llaisys::utils::nvidia::save_4d(out + idx, flo4_v);
    } else {
        for (int j = idx; j < numel; ++ j){
            const float t = llaisys::utils::nvidia::cast<float>(up[j]);
            const float s = llaisys::utils::nvidia::cast<float>(gate[j]);
            const float val = t * s / (1.0f + expf(-s));
            out[j] = llaisys::utils::nvidia::cast<T>(val);
        }
    }
}

template <typename T>
void swiglu_launch(std::byte *out, const std::byte *gate, const std::byte *up, const size_t &numel) {
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_gate = reinterpret_cast<const T *>(gate);
    const auto *d_up = reinterpret_cast<const T *>(up);

    dim3 blockDim(256);
    dim3 gridDim((((numel + 3) >> 2) + 255) >> 8);

    swiglu_kernel<<<gridDim, blockDim>>>(d_out, d_gate, d_up, numel);

    CUDA_CHECK(cudaGetLastError());
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
