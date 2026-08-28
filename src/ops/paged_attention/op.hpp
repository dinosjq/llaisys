#pragma once

#include "../../tensor/tensor.hpp"

namespace llaisys::ops {
// 自定义 paged_attention (FA-v2)
// tot_task_num: prefill = Σ ceil(seq_len/BLOCK_M)；decode = Σ 块数（flash_decoding 的 grid.x）
void paged_attention(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache, tensor_t block_ids, tensor_t cnt_idx, tensor_t tot_len, size_t tot_task_num, float scale, bool is_prefill, tensor_t attn_acc = nullptr, tensor_t attn_sum = nullptr, tensor_t attn_max = nullptr);
} // namespace llaisys::ops
