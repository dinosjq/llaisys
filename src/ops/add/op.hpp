#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 元素级相加：c = a + b。
// 要求：c、a、b 形状与数据类型一致，设备匹配。
void add(tensor_t c, tensor_t a, tensor_t b);
} // namespace llaisys::ops
