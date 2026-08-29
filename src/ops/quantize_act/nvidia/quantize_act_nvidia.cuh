#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void quantize_act(std::byte *out_q, std::byte *out_scale, const std::byte *in, llaisysDataType_t in_dtype,
                  size_t m, size_t k);
} // namespace llaisys::ops::nvidia
