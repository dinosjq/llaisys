#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 激活 per-token symmetric 量化（NVIDIA only）：in[m,k] FP → out_q[m,k] I8 + out_scale[m] F32。
// 独立步骤：先量化中间张量，再交给 linear_w8a8 消费。
void quantize_act(tensor_t out_q, tensor_t out_scale, tensor_t in);
} // namespace llaisys::ops
