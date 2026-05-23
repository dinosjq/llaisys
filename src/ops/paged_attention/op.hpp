#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 自定义 paged_attention (FA-v2)
void paged_attention(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache, tensor_t block_ids, size_t nblock, size_t totlen, float scale);
} // namespace llaisys::ops

