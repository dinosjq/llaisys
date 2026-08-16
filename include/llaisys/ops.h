#ifndef LLAISYS_OPS_H
#define LLAISYS_OPS_H

#include "tensor.h"

__C {
    __export void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b);
    __export void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals);
    __export void llaisysTopk(llaisysTensor_t out_idx, llaisysTensor_t out_val, llaisysTensor_t vals, size_t k);
    __export void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight);
    __export void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias);
    __export void llaisysKvCacheMove(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t block_ids_tensor, llaisysTensor_t cut_idx_tensor, llaisysTensor_t pos_ids_tensor, size_t max_seq_len);
    __export void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in);
    __export void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps);
    __export void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta);
    __export void llaisysROPEInvFreq(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids,
                                     llaisysTensor_t inv_freq);
    __export void llaisysPagedAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k_cache, llaisysTensor_t v_cache, llaisysTensor_t block_ids_tensor, llaisysTensor_t cut_idx_tensor, llaisysTensor_t tot_len_tensor, size_t max_seq_len, size_t tot_block_num, float scale, bool is_prefill, llaisysTensor_t attn_acc, llaisysTensor_t attn_sum, llaisysTensor_t attn_max);
    __export void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale);
    __export void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up);
}

#endif
