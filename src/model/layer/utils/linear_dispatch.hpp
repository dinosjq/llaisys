#pragma once

#include "../../../ops/linear/op.hpp"
#include "../../../ops/linear_w8a8/op.hpp"
#include "../../../tensor/tensor.hpp"

namespace llaisys::layers::utils {

// FP linear（scale 为空，in 为 FP），或 W8A8 量化 linear（scale 非空，in 为已量化 I8、a_scale 为 per-token scale）。
inline void linear_proj(tensor_t out, tensor_t in, tensor_t a_scale, tensor_t weight, tensor_t scale, tensor_t bias) {
    if (scale) {
        ops::linear_w8a8(out, in, a_scale, weight, scale, bias);
    } else {
        ops::linear(out, in, weight, bias);
    }
}

} // namespace llaisys::layers::utils
