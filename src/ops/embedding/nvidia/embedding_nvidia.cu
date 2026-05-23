#include "embedding_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <cstring>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *d_index, const std::byte *weight, const size_t numel, const size_t len, const size_t elem_size) {
    const size_t len_size = len * elem_size;

    const size_t byte_size = numel * sizeof(int64_t);
    std::byte *h_index = (std::byte *) malloc(byte_size);
    CUDA_CHECK(cudaMemcpy(h_index, d_index, byte_size, cudaMemcpyDeviceToHost));

    int64_t *index = reinterpret_cast<int64_t *>(h_index);

    for (size_t i = 0, o_offset = 0; i < numel; ++i, o_offset += len_size) {
        size_t idx = static_cast<size_t>(index[i]);
        size_t w_offset = idx * len_size;
        CUDA_CHECK(cudaMemcpy(out + o_offset, weight + w_offset, len_size, cudaMemcpyDeviceToDevice));
    }

    free(h_index);
}
} // namespace llaisys::ops::nvidia
