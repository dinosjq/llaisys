#include "quantize_w8_nvidia.cuh"

#include "../../../utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

__device__ __forceinline__ float warp_amax(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    }
    return v;
}

template <typename T>
__global__ void quantize_w8_kernel(int8_t *__restrict__ out_i8, float *__restrict__ scale,
                                   const T *__restrict__ weight, size_t n, size_t k) {
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) {
        return;
    }

    const T *row_w = weight + row * k;
    int8_t *row_q = out_i8 + row * k;

    float local_max = 0.f;
    for (size_t j = static_cast<size_t>(threadIdx.x); j < k; j += static_cast<size_t>(blockDim.x)) {
        local_max = fmaxf(local_max, fabsf(llaisys::utils::nvidia::cast<float>(row_w[j])));
    }
    local_max = warp_amax(local_max);

    __shared__ float warp_max[32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        warp_max[warp] = local_max;
    }
    __syncthreads();

    float amax = 0.f;
    if (threadIdx.x < ((blockDim.x + 31) >> 5)) {
        amax = warp_max[threadIdx.x];
    }
    if (threadIdx.x < 32) {
        amax = warp_amax(amax);
    }
    __shared__ float row_scale;
    if (threadIdx.x == 0) {
        row_scale = (amax == 0.f) ? 1.f : (amax / 127.f);
        scale[row] = row_scale;
    }
    __syncthreads();
    const float s = row_scale;
    const float inv = 1.f / s;

    for (size_t j = static_cast<size_t>(threadIdx.x); j < k; j += static_cast<size_t>(blockDim.x)) {
        const float q = nearbyintf(llaisys::utils::nvidia::cast<float>(row_w[j]) * inv);
        const int clipped = max(-128, min(127, static_cast<int>(q)));
        row_q[j] = static_cast<int8_t>(clipped);
    }
}

template <typename T>
void launch(std::byte *weight_i8, std::byte *scale, const std::byte *weight_fp, size_t n, size_t k) {
    const int threads = 256;
    quantize_w8_kernel<T><<<static_cast<int>(n), threads>>>(
        reinterpret_cast<int8_t *>(weight_i8), reinterpret_cast<float *>(scale), reinterpret_cast<const T *>(weight_fp),
        n, k);
}

} // namespace

namespace llaisys::ops::nvidia {

void quantize_w8(std::byte *weight_i8, std::byte *scale, const std::byte *weight_fp, llaisysDataType_t in_dtype,
                 size_t n, size_t k) {
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(weight_i8, scale, weight_fp, n, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(weight_i8, scale, weight_fp, n, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(weight_i8, scale, weight_fp, n, k);
    default:
        throw std::runtime_error("quantize_w8 nvidia: unsupported in dtype");
    }
}

} // namespace llaisys::ops::nvidia
