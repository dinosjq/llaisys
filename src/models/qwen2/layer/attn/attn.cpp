#include "attn.hpp"

#include "../../meta/qwen2_meta.hpp"
#include "../../weight/qwen2_w8_weight.hpp"
#include "../../../../model/layer/utils/linear_dispatch.hpp"
#include "../../../../ops/add/op.hpp"
#include "../../../../ops/kv_cache_move/op.hpp"
#include "../../../../ops/paged_attention/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/rope/op.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys::model {

Qwen2Attention::Qwen2Attention(qwen_attn_weights_t weights, size_t layer)
    : _w(std::move(weights)), _layer(layer) {}

void Qwen2Attention::forward(ModelContext &ctx) {
    forward(static_cast<Qwen2Context &>(ctx));
}

void Qwen2Attention::forward(Qwen2Context &ctx) {
    if (this->_w == nullptr || this->_layer >= ctx.k_caches.size() || this->_layer >= ctx.v_caches.size()) {
        throw std::invalid_argument("Qwen2Attention requires weights and per-layer KV caches");
    }

    auto &ws = ctx.workspace;
    const auto &meta = qwen_meta(ctx);
    const auto &w = *this->_w;
    const auto *w8 = dynamic_cast<const QwenW8AttnWeights *>(this->_w.get());
    const size_t layer = this->_layer;

    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t hs = meta.hs;
    const size_t token_count = ws.x->shape()[0];
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    ops::rms_norm(ws.x_norm, ws.x, w.norm_w, meta.epsilon);

    layers::utils::linear_proj(ws.q, ws.x_norm, w.q_w, w8 ? w8->q_scale : nullptr, w.q_b);
    layers::utils::linear_proj(ws.k, ws.x_norm, w.k_w, w8 ? w8->k_scale : nullptr, w.k_b);
    layers::utils::linear_proj(ws.v, ws.x_norm, w.v_w, w8 ? w8->v_scale : nullptr, w.v_b);

    ops::rope(ws.q_rope, ws.q->view({token_count, nh, dh}), ws.pos_ids, meta.theta);
    ops::rope(ws.k_rope, ws.k->view({token_count, nkvh, dh}), ws.pos_ids, meta.theta);

    ops::kv_cache_move(ctx.k_caches[layer], ws.k_rope, ws.block_ids, ws.cut_idx, ws.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer], ws.v->view({token_count, nkvh, dh}), ws.block_ids, ws.cut_idx, ws.pos_ids,
                       ctx.runtime.max_seq_len);

    ops::paged_attention(ws.attn_val, ws.q_rope, ctx.k_caches[layer], ctx.v_caches[layer], ws.block_ids, ws.cut_idx,
                         ws.tot_len, ctx.runtime.max_seq_len, ctx.runtime.tot_block_num, scale, ctx.runtime.is_prefill,
                         ws.attn_acc, ws.attn_sum, ws.attn_max);

    layers::utils::linear_proj(ws.attn_out, ws.attn_val->view({token_count, hs}), w.o_w, w8 ? w8->o_scale : nullptr, nullptr);

    ops::add(ws.x_attn, ws.x, ws.attn_out);
    std::swap(ws.x, ws.x_attn);
}

} // namespace llaisys::model
