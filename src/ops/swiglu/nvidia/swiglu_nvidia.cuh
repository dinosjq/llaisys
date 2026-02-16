#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t dtype, const size_t &numel);
}
