#include "quantize_w8_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_quant.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>

namespace {

// 权重量化 (量化前先做 K 维 Hadamard 旋转)
// 加载整行到 smem => 复用公共量化 (Hadamard 蝶形 + absmax => scale => round)
// 与 quantize_act 的激活旋转配套：x'·W' = (x·H)·(H·W) = x·W，输出不变但两侧动态范围均降低。
template <typename T>
__global__ void quantize_w8_kernel(int8_t *__restrict__ out_i8, 
                                   float *__restrict__ scale,
                                   const T *__restrict__ weight, 
                                   size_t n, 
                                   size_t k) 
{
    extern __shared__ float s_row[];
    const size_t row = static_cast<size_t>(blockIdx.x);
    if (row >= n) {
        return;
    }

    const T *row_w = weight + row * k;
    const size_t tid = static_cast<size_t>(threadIdx.x);

    // 1. 加载整行到 smem (k 必须为 128 的倍数)
    for (size_t j = tid; j < k; j += blockDim.x) {
        s_row[j] = llaisys::utils::nvidia::cast<float>(row_w[j]);
    }
    __syncthreads();

    // 2. 公共量化：Hadamard 旋转 + absmax => scale => round
    llaisys::utils::nvidia::quantize_smem_row(s_row, out_i8, scale, k, row, tid, blockDim.x);
}

template <typename T>
void launch(std::byte *weight_i8, std::byte *scale, const std::byte *weight_fp, size_t n, size_t k) {
    const int threads = 256;
    const size_t smem = k * sizeof(float);
    quantize_w8_kernel<T><<<static_cast<int>(n), threads, smem>>>(
        reinterpret_cast<int8_t *>(weight_i8), reinterpret_cast<float *>(scale), reinterpret_cast<const T *>(weight_fp),
        n, k);
    CUDA_CHECK(cudaGetLastError());
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
