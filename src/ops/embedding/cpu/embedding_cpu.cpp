#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, const size_t numel, const size_t len, const size_t elem_size){
    const size_t len_size = len * elem_size;
    const auto *_index = reinterpret_cast<const int64_t *>(index);
    size_t o_offset = 0;
    for (size_t i = 0; i < numel; ++i, o_offset += len_size) {
        size_t idx = static_cast<size_t>(_index[i]);
        size_t w_offset = idx * len_size;
        std::memcpy(out + o_offset, weight + w_offset, len_size);
    }
}
} // namespace llaisys::ops::cpu
