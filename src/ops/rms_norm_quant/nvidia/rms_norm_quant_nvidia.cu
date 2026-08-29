#include "rms_norm_quant_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_quant.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// 融合 rms_norm + Hadamard + 量化: 每 block 一行 (k 为 128 的倍数)
// rms_norm 公式与 FP 路径一致：inv = 1/sqrt(mean(x²)+eps), s_row = w·x·inv。
template <typename T>
__global__ void rms_norm_quant_kernel(int8_t *__restrict__ out_q, float *__restrict__ out_scale,
                                      const T *__restrict__ in, const T *__restrict__ weight, size_t k, float eps) {
    extern __shared__ float s_row[];
    const size_t row = blockIdx.x;
    const T *r_in = in + row * k;
    const size_t tid = threadIdx.x;

    // 1. rms_norm: 加载 + 平方和 => 归一 => 乘 weight => s_row
    float sum = 0.f;
    for (size_t j = tid; j < k; j += blockDim.x) {
        const float v = llaisys::utils::nvidia::cast<float>(r_in[j]);
        s_row[j] = v;
        sum += v * v;
    }
    __shared__ float sm_den[32];
    const size_t warp = tid >> 5;
    const int lane = static_cast<int>(tid & 31);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, off);
    }
    if (lane == 0) {
        sm_den[warp] = sum;
    }
    __syncthreads();
    if (tid < 32) {
        float den = sm_den[tid];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            den += __shfl_xor_sync(0xffffffffu, den, off);
        }
        if (tid == 0) {
            den = den / static_cast<float>(k) + eps;
            sm_den[0] = 1.0f / sqrtf(den);
        }
    }
    __syncthreads();
    const float inv = sm_den[0];
    for (size_t j = tid; j < k; j += blockDim.x) {
        s_row[j] *= inv * llaisys::utils::nvidia::cast<float>(weight[j]);
    }
    __syncthreads();

    // 2. 公共量化: Hadamard 旋转 + absmax => scale => round
    llaisys::utils::nvidia::quantize_smem_row(s_row, out_q, out_scale, k, row, tid, blockDim.x);
}

template <typename T>
void launch(std::byte *out_q, std::byte *out_scale, const std::byte *in, const std::byte *weight, size_t m, size_t k,
            float eps) {
    const size_t smem = k * sizeof(float);
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(rms_norm_quant_kernel<T>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                             static_cast<int>(smem));
    }
    rms_norm_quant_kernel<T><<<static_cast<int>(m), 1024, smem>>>(
        reinterpret_cast<int8_t *>(out_q), reinterpret_cast<float *>(out_scale), reinterpret_cast<const T *>(in),
        reinterpret_cast<const T *>(weight), k, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void rms_norm_quant(std::byte *out_q, std::byte *out_scale, const std::byte *in, const std::byte *weight,
                    llaisysDataType_t in_dtype, size_t m, size_t k, float eps) {
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out_q, out_scale, in, weight, m, k, eps);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out_q, out_scale, in, weight, m, k, eps);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out_q, out_scale, in, weight, m, k, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
    }
}

} // namespace llaisys::ops::nvidia
