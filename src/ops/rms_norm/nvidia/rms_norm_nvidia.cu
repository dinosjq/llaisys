#include "rms_norm_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

__device__ float warp_reduce(float val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__global__ void rms_norm_kernel(T *out, const T *in, const T *weight, const float eps, const size_t n, const size_t m){
    __shared__ float sm_den[8];
    extern __shared__ float sm_in[];

    const size_t row = blockIdx.x;

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    float sum = 0.0f;
    for (size_t col = tid; col < m; col += 256){
        float t = llaisys::utils::nvidia::cast<float>(in[row * m + col]);
        sm_in[col] = t;
        sum += t * t;
    }

    sum = warp_reduce(sum);

    if(lane_id == 0){
        sm_den[warp_id] = sum;
    }
    __syncthreads();

    if(tid < 32){
        float den = tid < 8 ? sm_den[tid] : 0.0f;
        den = warp_reduce(den);
        if(tid == 0){
            den /= m;
            den += eps;
            den = std::sqrt(den);
            sm_den[0] = den;
        }
    }

    __syncthreads();

    const float den = sm_den[0];
    for (size_t col = tid; col < m; col += 256) {
        const float w = llaisys::utils::nvidia::cast<float>(weight[col]);
        out[row * m + col] = llaisys::utils::nvidia::cast<T>(w * sm_in[col] / den);
    }
}

template <typename T>
void rms_norm_launch(std::byte *out, const std::byte *in, const std::byte *weight, const float &eps,
                     const size_t &n, const size_t &m) {
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_in = reinterpret_cast<const T *>(in);
    const auto *d_weight = reinterpret_cast<const T *>(weight);

    dim3 blockDim(256);
    dim3 gridDim(n);

    rms_norm_kernel<<<gridDim, blockDim, m * sizeof(float)>>>(d_out, d_in, d_weight, eps, n, m);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float &eps, llaisysDataType_t type,
              const size_t &n, const size_t &m) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_launch<float>(out, in, weight, eps, n, m);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_launch<llaisys::bf16_t>(out, in, weight, eps, n, m);
    case LLAISYS_DTYPE_F16:
        return rms_norm_launch<llaisys::fp16_t>(out, in, weight, eps, n, m);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
