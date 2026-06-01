#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 自定义缓存搬运算子
void kv_cache_move(tensor_t out, tensor_t in, tensor_t block_ids, tensor_t cut_idx, tensor_t pos_ids, size_t max_seq_len);
} // namespace llaisys::ops
