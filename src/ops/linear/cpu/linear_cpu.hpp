#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype, 
            const size_t &n, const size_t &m, const size_t &t);
void linear(std::byte *out, const std::byte *in, const std::byte *weight,  llaisysDataType_t dtype, const size_t &n, const size_t &m, const size_t &t);
}