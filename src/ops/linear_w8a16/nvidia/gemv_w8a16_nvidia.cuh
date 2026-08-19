#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

bool gemv_w8a16_supported(llaisysDataType_t dtype, size_t m, size_t n, size_t k);

void gemv_w8a16(std::byte *out, const std::byte *in, const std::byte *weight_i8, const std::byte *scale,
                const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t n, size_t k);

} // namespace llaisys::ops::nvidia
