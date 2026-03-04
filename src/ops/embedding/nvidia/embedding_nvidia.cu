#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cstring>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {
void embedding(std::byte *__restrict__ out, const std::byte *__restrict__ h_index, const std::byte *__restrict__ weight, const size_t numel, const size_t len, const size_t elem_size) {
    const size_t len_size = len * elem_size;

    const int64_t *index = reinterpret_cast<const int64_t *>(h_index);

    for (size_t i = 0, o_offset = 0; i < numel; ++i, o_offset += len_size) {
        size_t idx = static_cast<size_t>(index[i]);
        size_t w_offset = idx * len_size;
        CUDA_CHECK(cudaMemcpy(out + o_offset, weight + w_offset, len_size, cudaMemcpyDeviceToDevice));
    }
}
} // namespace llaisys::ops::nvidia
