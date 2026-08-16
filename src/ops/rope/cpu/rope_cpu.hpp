#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta, llaisysDataType_t dtype,
          const size_t &seqlen, const size_t &head, const size_t &d);
          
// inv_freq: float[d/2]，phi = pos * inv_freq[j]
void rope_inv_freq(std::byte *out, const std::byte *in, const std::byte *pos_ids, const std::byte *inv_freq,
                   llaisysDataType_t dtype, const size_t &seqlen, const size_t &head, const size_t &d);
} // namespace llaisys::ops::cpu
