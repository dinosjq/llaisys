#pragma once

#include "llaisys.h"
#include <cstddef>

namespace llaisys::ops::nvidia {

void dequant_w8(std::byte *out, const std::byte *weight_i8, const std::byte *scale, llaisysDataType_t out_dtype,
                size_t n, size_t k);

} // namespace llaisys::ops::nvidia
