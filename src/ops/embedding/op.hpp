#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 词嵌入查找：按 index 从 weight 取行拷贝到 out。
// 要求：index 为整数类型；out 与 weight 的嵌入维度一致，设备与数据类型匹配实现约束。
void embedding(tensor_t out, tensor_t index, tensor_t weight);
} // namespace llaisys::ops
