#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void topk(std::byte *out_idx, std::byte *out_val, const std::byte *vals, llaisysDataType_t type, size_t numel, size_t k);
}