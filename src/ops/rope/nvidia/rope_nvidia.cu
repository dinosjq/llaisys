#include "rope_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

template<typename T>
__global__ void rope_kernel(T *out, const T *in, const int64_t *pos_ids, const float theta, const size_t seqlen, const size_t head, const size_t d){
    extern __shared__ float smem[];

    const size_t i = blockIdx.y;
    const size_t h = blockIdx.x;
    const int64_t pi = pos_ids[i];

    const size_t tid = threadIdx.x;

    const size_t offset = ((i * head) + h) * d;
    for (size_t j = tid; j < d; j += blockDim.x){
        smem[j] = llaisys::utils::nvidia::cast<float>(in[offset + j]);
    }

    const float *in_a = smem;
    const float *in_b = in_a + (d >> 1);
    T *out_a = out + offset;
    T *out_b = out_a + (d >> 1);

    for (size_t j = tid; j < (d >> 1); j += blockDim.x) {
        const float den = std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(d));
        const float phi = static_cast<float>(pi) / den;
        const float cos_phi = std::cos(phi);
        const float sin_phi = std::sin(phi);

        out_a[j] = llaisys::utils::nvidia::cast<T>(in_a[j] * cos_phi - in_b[j] * sin_phi);
        out_b[j] = llaisys::utils::nvidia::cast<T>(in_b[j] * cos_phi + in_a[j] * sin_phi);
    }
}

template <typename T>
void rope_launch(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta,
                 const size_t &seqlen, const size_t &head, const size_t &d) {
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_in = reinterpret_cast<const T *>(in);
    const auto *d_pos_ids = reinterpret_cast<const int64_t *>(pos_ids);

    dim3 blockDim(128);
    dim3 gridDim(head, seqlen);

    rope_kernel<<<gridDim, blockDim, d * sizeof(float)>>>(d_out, d_in, d_pos_ids, theta, seqlen, head, d);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta, llaisysDataType_t type,
          const size_t &seqlen, const size_t &head, const size_t &d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_launch<float>(out, in, pos_ids, theta, seqlen, head, d);
    case LLAISYS_DTYPE_BF16:
        return rope_launch<llaisys::bf16_t>(out, in, pos_ids, theta, seqlen, head, d);
    case LLAISYS_DTYPE_F16:
        return rope_launch<llaisys::fp16_t>(out, in, pos_ids, theta, seqlen, head, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
