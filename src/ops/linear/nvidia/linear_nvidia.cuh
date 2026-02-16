#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype,
            const size_t m, const size_t n, const size_t k);
} // namespace llaisys::ops::nvidia