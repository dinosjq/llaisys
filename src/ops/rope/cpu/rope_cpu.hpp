#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta, llaisysDataType_t dtype, const size_t &seqlen, const size_t &head, const size_t &d);
}