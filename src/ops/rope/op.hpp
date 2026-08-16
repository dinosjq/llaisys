#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// RoPE（旋转位置编码）：将位置信息注入 in，结果写入 out。
// 入口 1：theta 为频率基，kernel 内计算 1/theta^(2j/d)。
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta);
// 入口 2：预计算 inv_freq[dh/2]（float），phi = pos * inv_freq[j]；无热路径 pow。
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, tensor_t inv_freq);
} // namespace llaisys::ops
