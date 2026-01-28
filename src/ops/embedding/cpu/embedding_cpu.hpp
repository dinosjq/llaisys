#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const int64_t *index, const std::byte *weight, llaisysDataType_t dtype, const size_t numel, const size_t len, const size_t elem_size);
}