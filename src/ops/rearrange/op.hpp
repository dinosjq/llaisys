#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 视图重排/拷贝：按 out 的 shape/stride 将 in 的数据布局复制到 out。
// 要求：out 与 in 数据类型一致，设备匹配；形状兼容对应视图定义。
void rearrange(tensor_t out, tensor_t in);
} // namespace llaisys::ops
