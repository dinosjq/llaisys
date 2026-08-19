#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// out[n,k] = weight_i8[n,k] * scale[n]  (per output row)
void dequant_w8(tensor_t out, tensor_t weight_i8, tensor_t scale);

} // namespace llaisys::ops
