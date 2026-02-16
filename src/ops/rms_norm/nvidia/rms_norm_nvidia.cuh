#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float &eps, llaisysDataType_t type,
              const size_t &n, const size_t &m);
} // namespace llaisys::ops::nvidia