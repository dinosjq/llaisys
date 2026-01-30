#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 线性变换：out = in @ weight^T + bias。
// 要求：矩阵乘维度可对齐；bias 可为可选同形状项，设备/类型需匹配实现约束。
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias);
} // namespace llaisys::ops
