#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {

bool gemv_supported(llaisysDataType_t dtype, size_t m, size_t n, size_t k);

void gemv(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
          llaisysDataType_t dtype, size_t m, size_t n, size_t k);

} // namespace llaisys::ops::nvidia
