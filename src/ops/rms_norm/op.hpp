#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// RMSNorm：按通道/特征做均方根归一化并乘以权重。
// 参数：eps 为数值稳定常数；要求 out 与 in 同形状与类型，weight 形状可广播匹配。
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps);
} // namespace llaisys::ops
