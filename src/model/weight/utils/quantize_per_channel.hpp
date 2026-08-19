#pragma once

#include "../../../ops/quantize_w8/op.hpp"
#include "../../../tensor/tensor.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::weight::utils {

// Allocate I8 + F32 scale on the same device as `weight`, then run ops::quantize_w8.
inline std::pair<tensor_t, tensor_t> quantize_weight_per_channel(const tensor_t &weight) {
    if (!weight || weight->ndim() != 2 || !weight->isContiguous()) {
        throw std::invalid_argument("quantize_weight_per_channel: need contiguous 2D weight");
    }
    const size_t n = weight->shape()[0];
    const size_t k = weight->shape()[1];
    tensor_t w_i8 = Tensor::create({n, k}, LLAISYS_DTYPE_I8, weight->deviceType(), weight->deviceId());
    tensor_t scale_t = Tensor::create({n}, LLAISYS_DTYPE_F32, weight->deviceType(), weight->deviceId());
    ops::quantize_w8(w_i8, scale_t, weight);
    return {std::move(w_i8), std::move(scale_t)};
}

} // namespace llaisys::weight::utils
