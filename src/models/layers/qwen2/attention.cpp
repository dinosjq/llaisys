#include "attention.hpp"

#include "../../../ops/add/op.hpp"
#include "../../../ops/kv_cache_move/op.hpp"
#include "../../../ops/linear/op.hpp"
#include "../../../ops/paged_attention/op.hpp"
#include "../../../ops/rms_norm/op.hpp"
#include "../../../ops/rope/op.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys::framework {

Qwen2Attention::Qwen2Attention(const AttnWeights *weights, size_t layer) : w_(weights), layer_(layer) {}

void Qwen2Attention::forward(ModelContext &ctx) {
    if (w_ == nullptr || layer_ >= ctx.k_caches.size() || layer_ >= ctx.v_caches.size()) {
        throw std::invalid_argument("Qwen2Attention requires weights and per-layer KV caches");
    }

    const size_t token_count = ctx.workspace.x->shape()[0];
    const float scale = 1.0f / std::sqrt(static_cast<float>(ctx.meta.dh));
    ops::rms_norm(ctx.workspace.x_norm, ctx.workspace.x, w_->norm_w, ctx.meta.epsilon);
    ops::linear(ctx.workspace.q, ctx.workspace.x_norm, w_->q_w, w_->q_b);
    ops::linear(ctx.workspace.k, ctx.workspace.x_norm, w_->k_w, w_->k_b);
    ops::linear(ctx.workspace.v, ctx.workspace.x_norm, w_->v_w, w_->v_b);
    ops::rope(ctx.workspace.q_rope, ctx.workspace.q->view({token_count, ctx.meta.nh, ctx.meta.dh}), ctx.workspace.pos_ids, ctx.meta.theta);
    ops::rope(ctx.workspace.k_rope, ctx.workspace.k->view({token_count, ctx.meta.nkvh, ctx.meta.dh}), ctx.workspace.pos_ids, ctx.meta.theta);
    ops::kv_cache_move(ctx.k_caches[layer_], ctx.workspace.k_rope, ctx.workspace.block_ids, ctx.workspace.cut_idx, ctx.workspace.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer_], ctx.workspace.v->view({token_count, ctx.meta.nkvh, ctx.meta.dh}), ctx.workspace.block_ids, ctx.workspace.cut_idx, ctx.workspace.pos_ids,
                       ctx.runtime.max_seq_len);
    ops::paged_attention(ctx.workspace.attn_val, ctx.workspace.q_rope, ctx.k_caches[layer_], ctx.v_caches[layer_], ctx.workspace.block_ids, ctx.workspace.cut_idx,
                          ctx.workspace.tot_len, ctx.runtime.max_seq_len, ctx.k_caches[layer_]->shape()[0], scale, ctx.runtime.is_prefill, ctx.workspace.attn_acc,
                          ctx.workspace.attn_sum, ctx.workspace.attn_max);
    ops::linear(ctx.workspace.attn_out, ctx.workspace.attn_val->view({token_count, ctx.meta.hs}), w_->o_w, nullptr);
    ops::add(ctx.workspace.x_attn, ctx.workspace.x, ctx.workspace.attn_out);
    std::swap(ctx.workspace.x, ctx.workspace.x_attn);
}

} // namespace llaisys::framework
