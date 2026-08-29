#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {

// 融合 swiglu + Hadamard 旋转 + per-token 量化（NVIDIA only）：
// out_q[m,k] I8 + out_scale[m] F32 = quantize(swiglu(gate, up)·H)。
// W8A8 路径用它替代「swiglu + quantize_act」两步，让 gate/up 融合后直接输出量化精度。
void swiglu_quant(tensor_t out_q, tensor_t out_scale, tensor_t gate, tensor_t up);

} // namespace llaisys::ops
