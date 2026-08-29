#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

void rms_norm_quant(std::byte *out_q, std::byte *out_scale, const std::byte *in, const std::byte *weight,
                    llaisysDataType_t in_dtype, size_t m, size_t k, float eps);

} // namespace llaisys::ops::nvidia
