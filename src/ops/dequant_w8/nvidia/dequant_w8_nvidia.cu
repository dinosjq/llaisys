#include "dequant_w8_nvidia.cuh"

#include "../../../utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

template <typename T>
__global__ void dequant_w8_kernel(T *__restrict__ out, const int8_t *__restrict__ weight,
                                  const float *__restrict__ scale, size_t n, size_t k) {
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t numel = n * k;
    if (idx >= numel) {
        return;
    }
    const size_t row = idx / k;
    const float v = static_cast<float>(weight[idx]) * scale[row];
    out[idx] = llaisys::utils::nvidia::cast<T>(v);
}

template <typename T>
void launch(std::byte *out, const std::byte *weight_i8, const std::byte *scale, size_t n, size_t k) {
    const size_t numel = n * k;
    const dim3 block(256);
    const dim3 grid((numel + 255) / 256);
    dequant_w8_kernel<T><<<grid, block>>>(reinterpret_cast<T *>(out), reinterpret_cast<const int8_t *>(weight_i8),
                                          reinterpret_cast<const float *>(scale), n, k);
}

} // namespace

namespace llaisys::ops::nvidia {

void dequant_w8(std::byte *out, const std::byte *weight_i8, const std::byte *scale, llaisysDataType_t out_dtype,
                size_t n, size_t k) {
    switch (out_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out, weight_i8, scale, n, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out, weight_i8, scale, n, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out, weight_i8, scale, n, k);
    default:
        throw std::runtime_error("dequant_w8 nvidia: unsupported out dtype");
    }
}

} // namespace llaisys::ops::nvidia
