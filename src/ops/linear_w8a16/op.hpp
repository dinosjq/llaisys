#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// out[m,n] = scale[n] * (in[m,k] @ weight_i8[n,k]^T) + bias[n]?
void linear_w8a16(tensor_t out, tensor_t in, tensor_t weight_i8, tensor_t scale, tensor_t bias);

} // namespace llaisys::ops
