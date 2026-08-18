#include "attn.hpp"

#include "../../../qwen2/meta/qwen2_meta.hpp"
#include "../../../../ops/add/op.hpp"
#include "../../../../ops/kv_cache_move/op.hpp"
#include "../../../../ops/linear/op.hpp"
#include "../../../../ops/paged_attention/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/rope/op.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys::model {

LlamaAttention::LlamaAttention(llama_attn_weights_t weights, size_t layer)
    : _w(std::move(weights)), _layer(layer) {}

void LlamaAttention::forward(ModelContext &ctx) {
    forward(static_cast<LlamaContext &>(ctx));
}

void LlamaAttention::forward(LlamaContext &ctx) {
    if (this->_w == nullptr || this->_layer >= ctx.k_caches.size() || this->_layer >= ctx.v_caches.size()) {
        throw std::invalid_argument("LlamaAttention requires weights and per-layer KV caches");
    }

    auto &ws = ctx.workspace;
    const auto &meta = qwen_meta(ctx);
    const auto &w = *this->_w;
    const size_t layer = this->_layer;

    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t hs = meta.hs;
    const size_t token_count = ws.x->shape()[0];
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    ops::rms_norm(ws.x_norm, ws.x, w.norm_w, meta.epsilon);

    ops::linear(ws.q, ws.x_norm, w.q_w, nullptr);
    ops::linear(ws.k, ws.x_norm, w.k_w, nullptr);
    ops::linear(ws.v, ws.x_norm, w.v_w, nullptr);

    ops::rope(ws.q_rope, ws.q->view({token_count, nh, dh}), ws.pos_ids, ctx.inv_freq);
    ops::rope(ws.k_rope, ws.k->view({token_count, nkvh, dh}), ws.pos_ids, ctx.inv_freq);

    ops::kv_cache_move(ctx.k_caches[layer], ws.k_rope, ws.block_ids, ws.cut_idx, ws.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer], ws.v->view({token_count, nkvh, dh}), ws.block_ids, ws.cut_idx, ws.pos_ids,
                       ctx.runtime.max_seq_len);

    ops::paged_attention(ws.attn_val, ws.q_rope, ctx.k_caches[layer], ctx.v_caches[layer], ws.block_ids, ws.cut_idx,
                         ws.tot_len, ctx.runtime.max_seq_len, ctx.runtime.tot_block_num, scale, ctx.runtime.is_prefill,
                         ws.attn_acc, ws.attn_sum, ws.attn_max);

    ops::linear(ws.attn_out, ws.attn_val->view({token_count, hs}), w.o_w, nullptr);

    ops::add(ws.x_attn, ws.x, ws.attn_out);
    std::swap(ws.x, ws.x_attn);
}

} // namespace llaisys::model
