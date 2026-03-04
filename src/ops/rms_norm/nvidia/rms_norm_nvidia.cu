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

template <typename T>
__global__ void rms_norm_kernel(T *__restrict__ out, const T *__restrict__ in, const T *__restrict__ weight,
                                const float eps, const size_t n, const size_t m) {
    __shared__ float sm_den[8];
    extern __shared__ float sm_in[];

    const size_t row = blockIdx.x;
    const T *r_in = in + row * m;
    T *r_out = out + row * m;

    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    float sum = 0.0f;
    for (size_t col = (tid << 2); col < m; col += 1024) {
        if (col + 3 < m) {
            const float4 flo4 = llaisys::utils::nvidia::load_4d(r_in + col);
            sm_in[col | 0] = flo4.x;
            sm_in[col | 1] = flo4.y;
            sm_in[col | 2] = flo4.z;
            sm_in[col | 3] = flo4.w;

            sum += flo4.x * flo4.x;
            sum += flo4.y * flo4.y;
            sum += flo4.z * flo4.z;
            sum += flo4.w * flo4.w;
        } else {
            for (size_t j = col; j < m; ++j) {
                const float t = llaisys::utils::nvidia::cast<float>(r_in[j]);
                sm_in[j] = t;
                sum += t * t;
            }
        }
    }

    sum = warp_reduce(sum);

    if (lane_id == 0) {
        sm_den[warp_id] = sum;
    }
    __syncthreads();

    if (tid < 32) {
        float den = tid < 8 ? sm_den[tid] : 0.0f;
        den = warp_reduce(den);
        if (tid == 0) {
            den /= m;
            den += eps;
            den = sqrtf(den);
            sm_den[0] = den;
        }
    }

    __syncthreads();

    const float inv = 1.0f / sm_den[0];
    // const float den = sm_den[0];

    for (size_t col = (tid << 2); col < m; col += 1024) {
        if (col + 3 < m) {
            const float4 w_flo4 = llaisys::utils::nvidia::load_4d(weight + col);
            const float4 flo4 = make_float4(
                w_flo4.x * sm_in[col | 0] * inv,
                w_flo4.y * sm_in[col | 1] * inv,
                w_flo4.z * sm_in[col | 2] * inv,
                w_flo4.w * sm_in[col | 3] * inv
            );
            llaisys::utils::nvidia::save_4d(r_out + col, flo4);
        } else {
            for (size_t j = col; j < m; ++j) {
                const float w = llaisys::utils::nvidia::cast<float>(weight[j]);
                const float val = w * sm_in[j] * inv;
                r_out[j] = llaisys::utils::nvidia::cast<T>(val);
            }
        }
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
