#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// SwiGLU 激活：out = SiLU(gate) * up。
// 要求：gate、up 与 out 形状一致，类型/设备匹配实现约束。
void swiglu(tensor_t out, tensor_t gate, tensor_t up);
} // namespace llaisys::ops
