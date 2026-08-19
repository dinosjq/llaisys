#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void quantize_w8(std::byte *weight_i8, std::byte *scale, const std::byte *weight_fp, llaisysDataType_t in_dtype,
                 size_t n, size_t k);

} // namespace llaisys::ops::nvidia
