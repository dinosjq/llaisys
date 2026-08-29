#include <cstddef>

#include "llaisys/ops.h"
#include "llaisys_tensor.hpp"

#include "../ops/add/op.hpp"
#include "../ops/argmax/op.hpp"
#include "../ops/embedding/op.hpp"
#include "../ops/kv_cache_move/op.hpp"
#include "../ops/linear/op.hpp"
#include "../ops/dequant_w8/op.hpp"
#include "../ops/quantize_w8/op.hpp"
#include "../ops/linear_w8a8/op.hpp"
#include "../ops/quantize_act/op.hpp"
#include "../ops/paged_attention/op.hpp"
#include "../ops/rearrange/op.hpp"
#include "../ops/rms_norm/op.hpp"
#include "../ops/rms_norm_quant/op.hpp"
#include "../ops/rope/op.hpp"
#include "../ops/self_attention/op.hpp"
#include "../ops/swiglu/op.hpp"
#include "../ops/swiglu_quant/op.hpp"
#include "../ops/top_k/op.hpp"

__C {
void llaisysAdd(llaisysTensor_t c, llaisysTensor_t a, llaisysTensor_t b) {
    llaisys::ops::add(c->tensor, a->tensor, b->tensor);
}

void llaisysArgmax(llaisysTensor_t max_idx, llaisysTensor_t max_val, llaisysTensor_t vals) {
    llaisys::ops::argmax(max_idx->tensor, max_val->tensor, vals->tensor);
}

void llaisysTopk(llaisysTensor_t out_idx, llaisysTensor_t out_val, llaisysTensor_t vals, size_t k) {
    llaisys::ops::topk(out_idx->tensor, out_val->tensor, vals->tensor, k);
}

void llaisysEmbedding(llaisysTensor_t out, llaisysTensor_t index, llaisysTensor_t weight) {
    llaisys::ops::embedding(out->tensor, index->tensor, weight->tensor);
}

void llaisysLinear(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, llaisysTensor_t bias) {
    llaisys::ops::linear(out->tensor, in->tensor, weight->tensor, bias ? bias->tensor : nullptr);
}

void llaisysDequantW8(llaisysTensor_t out, llaisysTensor_t weight_i8, llaisysTensor_t scale) {
    llaisys::ops::dequant_w8(out->tensor, weight_i8->tensor, scale->tensor);
}

void llaisysQuantizeW8(llaisysTensor_t weight_i8, llaisysTensor_t scale, llaisysTensor_t weight_fp) {
    llaisys::ops::quantize_w8(weight_i8->tensor, scale->tensor, weight_fp->tensor);
}

void llaisysQuantizeAct(llaisysTensor_t out_q, llaisysTensor_t out_scale, llaisysTensor_t in) {
    llaisys::ops::quantize_act(out_q->tensor, out_scale->tensor, in->tensor);
}

void llaisysLinearW8A8(llaisysTensor_t out, llaisysTensor_t in_q, llaisysTensor_t a_scale, llaisysTensor_t weight_i8,
                       llaisysTensor_t w_scale, llaisysTensor_t bias) {
    llaisys::ops::linear_w8a8(out->tensor, in_q->tensor, a_scale->tensor, weight_i8->tensor, w_scale->tensor,
                              bias ? bias->tensor : nullptr);
}

void llaisysKvCacheMove(llaisysTensor_t out,
                        llaisysTensor_t in,
                        llaisysTensor_t block_ids_tensor,
                        llaisysTensor_t cut_idx_tensor,
                        llaisysTensor_t pos_ids_tensor,
                        llaisysTensor_t scale_tensor) {
    llaisys::ops::kv_cache_move(out->tensor,
                                in->tensor,
                                block_ids_tensor->tensor,
                                cut_idx_tensor->tensor,
                                pos_ids_tensor->tensor,
                                scale_tensor ? scale_tensor->tensor : nullptr);
}

void llaisysRearrange(llaisysTensor_t out, llaisysTensor_t in) {
    llaisys::ops::rearrange(out->tensor, in->tensor);
}

void llaisysRmsNorm(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t weight, float eps) {
    llaisys::ops::rms_norm(out->tensor, in->tensor, weight->tensor, eps);
}

void llaisysRmsNormQuant(llaisysTensor_t out_q, llaisysTensor_t out_scale, llaisysTensor_t in,
                         llaisysTensor_t weight, float eps) {
    llaisys::ops::rms_norm_quant(out_q->tensor, out_scale->tensor, in->tensor, weight->tensor, eps);
}

void llaisysROPE(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, float theta) {
    llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, theta);
}

void llaisysROPEInvFreq(llaisysTensor_t out, llaisysTensor_t in, llaisysTensor_t pos_ids, llaisysTensor_t inv_freq) {
    llaisys::ops::rope(out->tensor, in->tensor, pos_ids->tensor, inv_freq->tensor);
}

void llaisysPagedAttention(llaisysTensor_t attn_val,
                           llaisysTensor_t q,
                           llaisysTensor_t k_cache,
                           llaisysTensor_t v_cache,
                           llaisysTensor_t block_ids_tensor,
                           llaisysTensor_t cut_idx_tensor,
                           llaisysTensor_t tot_len_tensor,
                           size_t tot_task_num,
                           float scale,
                           bool is_prefill,
                           llaisysTensor_t attn_acc,
                           llaisysTensor_t attn_sum,
                           llaisysTensor_t attn_max,
                           llaisysTensor_t k_scale,
                           llaisysTensor_t v_scale)
{
    llaisys::ops::paged_attention(attn_val->tensor,
                                   q->tensor,
                                   k_cache->tensor,
                                   v_cache->tensor,
                                   block_ids_tensor->tensor,
                                   cut_idx_tensor->tensor,
                                   tot_len_tensor->tensor,
                                   tot_task_num,
                                   scale,
                                   is_prefill,
                                   attn_acc ? attn_acc->tensor : nullptr,
                                   attn_sum ? attn_sum->tensor : nullptr,
                                   attn_max ? attn_max->tensor : nullptr,
                                   k_scale ? k_scale->tensor : nullptr,
                                   v_scale ? v_scale->tensor : nullptr);
}

void llaisysSelfAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k, llaisysTensor_t v, float scale) {
    llaisys::ops::self_attention(attn_val->tensor, q->tensor, k->tensor, v->tensor, scale);
}

void llaisysSwiGLU(llaisysTensor_t out, llaisysTensor_t gate, llaisysTensor_t up) {
    llaisys::ops::swiglu(out->tensor, gate->tensor, up->tensor);
}

void llaisysSwiGLUQuant(llaisysTensor_t out_q, llaisysTensor_t out_scale, llaisysTensor_t gate, llaisysTensor_t up) {
    llaisys::ops::swiglu_quant(out_q->tensor, out_scale->tensor, gate->tensor, up->tensor);
}
}
