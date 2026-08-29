#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// 融合 rms_norm + Hadamard 旋转 + per-token 量化（NVIDIA only）：
// out_q[m,k] I8 + out_scale[m] F32 = quantize(rms_norm(in, weight, eps)·H)。
// W8A8 路径用它替代「rms_norm + quantize_act」两步，让 norm 输出直接是量化精度。
void rms_norm_quant(tensor_t out_q, tensor_t out_scale, tensor_t in, tensor_t weight, float eps);

} // namespace llaisys::ops
