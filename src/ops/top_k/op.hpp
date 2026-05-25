#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 求 top_k 大的值及其索引
void topk(tensor_t out_idx, tensor_t out_val, tensor_t vals, size_t k);
} // namespace llaisys::ops
