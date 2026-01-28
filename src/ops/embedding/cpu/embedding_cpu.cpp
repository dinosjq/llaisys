#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight, llaisysDataType_t dtype, const size_t numel, const size_t len, const size_t elem_size){
    const size_t len_size = len * elem_size;
    size_t o_offset = 0;
    for (size_t i = 0; i < numel; ++i, o_offset += len_size) {
        size_t idx = static_cast<size_t>(index[i]);
        size_t w_offset = idx * len_size;
        std::memcpy(out + o_offset, weight + w_offset, len_size);
    }
}
} // namespace llaisys::ops::cpu
