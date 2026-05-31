#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace llaisys::ops::nvidia {

template <typename T>
__global__ void embedding_gather_kernel(T *out, const int64_t *index, const T *weight, size_t numel, size_t len) {
    size_t row = static_cast<size_t>(blockIdx.x) + static_cast<size_t>(blockIdx.y) * gridDim.x;
    if (row >= numel) return;

    size_t idx = static_cast<size_t>(index[row]);
    const T *src = weight + idx * len;
    T *dst = out + row * len;

    constexpr size_t VEC = 4;
    const size_t vec_len = len & ~(VEC - 1);

    for (size_t i = static_cast<size_t>(threadIdx.x) * VEC; i + (VEC - 1) < vec_len; i += static_cast<size_t>(blockDim.x) * VEC) {
        const float4 v = llaisys::utils::nvidia::load_4d(src + i);
        llaisys::utils::nvidia::save_4d(dst + i, v);
    }

    for (size_t i = vec_len + threadIdx.x; i < len; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <typename T>
void embedding_launch(std::byte *out, const std::byte *d_index, const std::byte *weight, const size_t numel, const size_t len) {
    if (numel == 0 || len == 0) return;

    const int64_t *index = reinterpret_cast<const int64_t *>(d_index);
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_weight = reinterpret_cast<const T *>(weight);

    const unsigned int MAX_X = 65535u;
    size_t blocks_x = std::min(numel, (size_t)MAX_X);
    size_t blocks_y = (numel + blocks_x - 1) / blocks_x;
    dim3 grid((unsigned)blocks_x, (unsigned)blocks_y);

    int threads = 256;
    if (len < (size_t)threads) threads = (int)len;
    if (threads <= 0) threads = 1;

    embedding_gather_kernel<<<grid, threads>>>(d_out, index, d_weight, numel, len);
    CUDA_CHECK(cudaGetLastError());
}

void embedding(std::byte *out, const std::byte *d_index, const std::byte *weight, llaisysDataType_t dtype,
               const size_t numel, const size_t len) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_launch<float>(out, d_index, weight, numel, len);
    case LLAISYS_DTYPE_BF16:
        return embedding_launch<llaisys::bf16_t>(out, d_index, weight, numel, len);
    case LLAISYS_DTYPE_F16:
        return embedding_launch<llaisys::fp16_t>(out, d_index, weight, numel, len);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
