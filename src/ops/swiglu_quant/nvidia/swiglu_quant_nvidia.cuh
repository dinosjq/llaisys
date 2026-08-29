#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void swiglu_quant(std::byte *out_q, std::byte *out_scale, const std::byte *gate, const std::byte *up,
                  llaisysDataType_t in_dtype, size_t m, size_t k);

} // namespace llaisys::ops::nvidia
