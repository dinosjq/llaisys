#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 自注意力：用 q 与 k 计算注意力权重并加权汇聚 v，写入 attn_val。
// 参数：scale 为打分缩放系数；要求张量形状满足 (T, H, Dh) 约定且类型/设备一致。
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale);
} // namespace llaisys::ops
