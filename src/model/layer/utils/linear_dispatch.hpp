#pragma once

#include "../../../ops/linear/op.hpp"
#include "../../../ops/linear_w8a16/op.hpp"
#include "../../../tensor/tensor.hpp"

namespace llaisys::layers::utils {

// FP linear, or W8A16 when scale is non-null.
inline void linear_proj(tensor_t out, tensor_t in, tensor_t weight, tensor_t scale, tensor_t bias) {
    if (scale) {
        ops::linear_w8a16(out, in, weight, scale, bias);
    } else {
        ops::linear(out, in, weight, bias);
    }
}

} // namespace llaisys::layers::utils
