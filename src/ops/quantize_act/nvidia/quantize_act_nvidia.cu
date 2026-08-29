#include "quantize_act_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_quant.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace {

// 激活 per-token symmetric 量化 (量化前先做 K 维 Hadamard 旋转)
// 加载整行到 smem => 复用公共量化 (Hadamard 蝶形 + absmax => scale => round)
template <typename T>
__global__ void quantize_act_kernel(int8_t *__restrict__ out_q,
                                    float *__restrict__ out_scale,
                                    const T *__restrict__ in, 
                                    size_t k) {
    extern __shared__ float s_row[];
    const size_t row = blockIdx.x;
    const T *row_in = in + row * k;
    const size_t tid = threadIdx.x;

    // 1. 加载整行到 smem (k 必须为 128 的倍数)
    for (size_t j = tid; j < k; j += blockDim.x) {
        s_row[j] = llaisys::utils::nvidia::cast<float>(row_in[j]);
    }
    __syncthreads();

    // 2. 公共量化: Hadamard 旋转 + absmax => scale => round
    llaisys::utils::nvidia::quantize_smem_row(s_row, out_q, out_scale, k, row, tid, blockDim.x);
}

template <typename T>
void launch(std::byte *out_q, std::byte *out_scale, const std::byte *in, size_t m, size_t k) {
    const size_t smem = k * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(quantize_act_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
    }
    quantize_act_kernel<T><<<static_cast<int>(m), 1024, smem>>>(
        reinterpret_cast<int8_t *>(out_q), reinterpret_cast<float *>(out_scale), reinterpret_cast<const T *>(in), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void quantize_act(std::byte *out_q, std::byte *out_scale, const std::byte *in, llaisysDataType_t in_dtype,
                  size_t m, size_t k) {
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out_q, out_scale, in, m, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out_q, out_scale, in, m, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out_q, out_scale, in, m, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
    }
}

} // namespace llaisys::ops::nvidia
