#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 求最大值及其索引：从 vals 按最后一维计算。
// 输出：max_val 保存最大值，max_idx 保存对应索引；形状需匹配约定，设备/类型受实现限制。
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals);
} // namespace llaisys::ops
