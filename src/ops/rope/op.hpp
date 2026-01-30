#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// RoPE（旋转位置编码）：将位置信息注入 in，结果写入 out。
// 参数：pos_ids 为位置索引，theta 为频率基；要求 out/in 形状与类型匹配实现约定。
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
} // namespace llaisys::ops
