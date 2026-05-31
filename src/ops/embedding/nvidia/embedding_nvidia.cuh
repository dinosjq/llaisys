#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t dtype,
			   const size_t numel, const size_t len);
}