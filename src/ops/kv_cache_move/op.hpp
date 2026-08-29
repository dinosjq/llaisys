#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 自定义缓存搬运算子
// 量化模式：out 为 I8（{block_num,block_size,nkvh,dh}），scale 输出 F32（{block_num,block_size,nkvh}）；
// 非量化模式：out/in 同 FP dtype，scale 传 nullptr。
void kv_cache_move(tensor_t out, tensor_t in, tensor_t block_ids, tensor_t cut_idx, tensor_t pos_ids, tensor_t scale = nullptr);
} // namespace llaisys::ops
