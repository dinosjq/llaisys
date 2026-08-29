#include "swiglu_quant_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_quant.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// 融合 swiglu + Hadamard + 量化：每 block 一行（k 为 128 的倍数）。
// swiglu 公式与 FP 路径一致：out = up·gate / (1+exp(-gate))。
template <typename T>
__global__ void swiglu_quant_kernel(int8_t *__restrict__ out_q, float *__restrict__ out_scale,
                                    const T *__restrict__ gate, const T *__restrict__ up, size_t k) {
    extern __shared__ float s_row[];
    const size_t row = blockIdx.x;
    const T *g = gate + row * k;
    const T *u = up + row * k;
    const size_t tid = threadIdx.x;

    // 1. swiglu => s_row
    for (size_t j = tid; j < k; j += blockDim.x) {
        const float s = llaisys::utils::nvidia::cast<float>(g[j]);
        const float t = llaisys::utils::nvidia::cast<float>(u[j]);
        s_row[j] = t * s / (1.0f + expf(-s));
    }
    __syncthreads();

    // 2. 公共量化：Hadamard 旋转 + absmax => scale => round
    llaisys::utils::nvidia::quantize_smem_row(s_row, out_q, out_scale, k, row, tid, blockDim.x);
}

template <typename T>
void launch(std::byte *out_q, std::byte *out_scale, const std::byte *gate, const std::byte *up, size_t m, size_t k) {
    const size_t smem = k * sizeof(float);
    swiglu_quant_kernel<T><<<static_cast<int>(m), 256, smem>>>(
        reinterpret_cast<int8_t *>(out_q), reinterpret_cast<float *>(out_scale), reinterpret_cast<const T *>(gate),
        reinterpret_cast<const T *>(up), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void swiglu_quant(std::byte *out_q, std::byte *out_scale, const std::byte *gate, const std::byte *up,
                  llaisysDataType_t in_dtype, size_t m, size_t k) {
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out_q, out_scale, gate, up, m, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out_q, out_scale, gate, up, m, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out_q, out_scale, gate, up, m, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
    }
}

} // namespace llaisys::ops::nvidia
