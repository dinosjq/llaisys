#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <algorithm>
#include <type_traits>

namespace llaisys::ops::nvidia {

constexpr int WARP_SIZE = 32;
constexpr int WARP_PER_BLOCK = 8;
constexpr int BLOCK_DIM = WARP_SIZE * WARP_PER_BLOCK;

template <typename T>
__global__ void embedding_gather_kernel(T *out, const int64_t *index, const T *weight, size_t numel, size_t len, const int padding) {
    const size_t tid = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;

    const size_t out_row = blockIdx.x * WARP_PER_BLOCK + warp_id;

    if (out_row >= numel) return; 

    size_t weight_row = static_cast<size_t>(index[out_row] + padding);

    const T *src = weight + weight_row * len;
    T *dst = out + out_row * len;

    for (size_t i = lane_id << 2; i < len; i += 128) {
        if (i + 3 < len) {
            llaisys::utils::nvidia::copy_4d(src + i, dst + i);
        } else {
            for (size_t j = 0; j < 4; ++ j) {
                if(i + j < len) {
                    dst[i + j] = src[i + j];
                }
            }
        }
    }
}

template <typename T>
void embedding_launch(std::byte *out, const std::byte *d_index, const std::byte *weight, const size_t numel, const size_t len, const int padding) {
    if (numel == 0 || len == 0) return;

    auto *index = reinterpret_cast<const int64_t *>(d_index);
    auto *d_out = reinterpret_cast<T *>(out);
    auto *d_weight = reinterpret_cast<const T *>(weight);

    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim((numel + WARP_PER_BLOCK - 1) / WARP_PER_BLOCK);

    embedding_gather_kernel<<<gridDim, blockDim>>>(d_out, index, d_weight, numel, len, padding);
}

void embedding(std::byte *out, const std::byte *d_index, const std::byte *weight, llaisysDataType_t dtype,
               const size_t numel, const size_t len, const int padding) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return embedding_launch<float>(out, d_index, weight, numel, len, padding);
    case LLAISYS_DTYPE_BF16:
        return embedding_launch<llaisys::bf16_t>(out, d_index, weight, numel, len, padding);
    case LLAISYS_DTYPE_F16:
        return embedding_launch<llaisys::fp16_t>(out, d_index, weight, numel, len, padding);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}

} // namespace llaisys::ops::nvidia
