#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float &eps, llaisysDataType_t dtype,
            const size_t &n, const size_t &m);
} // namespace llaisys::ops::cpu