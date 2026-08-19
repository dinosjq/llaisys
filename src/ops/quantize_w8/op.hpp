#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// Symmetric per-row quant: weight_fp[n,k] → weight_i8[n,k] + scale_f32[n]
// scale[i] = max_j |W[i,j]| / 127 (or 1 if row is zero).
void quantize_w8(tensor_t weight_i8, tensor_t scale, tensor_t weight_fp);

} // namespace llaisys::ops
