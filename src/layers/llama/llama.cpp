#include "llama.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/kv_cache_move/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/paged_attention/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/swiglu/op.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys::framework {

void LlamaEmbedding::forward(ModelContext &ctx) {
    auto &ws = ctx.workspace;
    auto &runtime = ctx.runtime;

    auto token_ids = Tensor::create({runtime.token_ids.size()}, LLAISYS_DTYPE_I64, ws.x->deviceType(), ws.x->deviceId());
    token_ids->load(runtime.token_ids.data());
    ops::embedding(ws.x, token_ids, ctx.in_embed);
}

LlamaAttention::LlamaAttention(const AttnWeights *weights, size_t layer) : w_(weights), layer_(layer) {}

void LlamaAttention::forward(ModelContext &ctx) {
    if (w_ == nullptr || layer_ >= ctx.k_caches.size() || layer_ >= ctx.v_caches.size()) {
        throw std::invalid_argument("LlamaAttention requires weights and per-layer KV caches");
    }

    auto &ws = ctx.workspace;
    const auto &meta = ctx.meta;
    const auto &w = *w_;

    const size_t token_count = ws.x->shape()[0];
    const float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));

    ops::rms_norm(ws.x_norm, ws.x, w.norm_w, meta.epsilon);

    // Llama: attention_bias=false → pass nullptr biases
    ops::linear(ws.q, ws.x_norm, w.q_w, nullptr);
    ops::linear(ws.k, ws.x_norm, w.k_w, nullptr);
    ops::linear(ws.v, ws.x_norm, w.v_w, nullptr);

    // Note: Llama-3.2 rope_scaling (llama3) is not applied yet; uses meta.theta only.
    ops::rope(ws.q_rope, ws.q->view({token_count, meta.nh, meta.dh}), ws.pos_ids, meta.theta);
    ops::rope(ws.k_rope, ws.k->view({token_count, meta.nkvh, meta.dh}), ws.pos_ids, meta.theta);

    ops::kv_cache_move(ctx.k_caches[layer_], ws.k_rope, ws.block_ids, ws.cut_idx, ws.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer_], ws.v->view({token_count, meta.nkvh, meta.dh}), ws.block_ids, ws.cut_idx, ws.pos_ids,
                       ctx.runtime.max_seq_len);

    ops::paged_attention(ws.attn_val, ws.q_rope, ctx.k_caches[layer_], ctx.v_caches[layer_], ws.block_ids, ws.cut_idx, ws.tot_len,
                         ctx.runtime.max_seq_len, ctx.runtime.tot_block_num, scale, ctx.runtime.is_prefill, ws.attn_acc, ws.attn_sum,
                         ws.attn_max);

    ops::linear(ws.attn_out, ws.attn_val->view({token_count, meta.hs}), w.o_w, nullptr);
    ops::add(ws.x_attn, ws.x, ws.attn_out);
    std::swap(ws.x, ws.x_attn);
}

LlamaFFN::LlamaFFN(const FfnWeights *weights, size_t layer_index_for_debug) : w_(weights), layer_index_for_debug_(layer_index_for_debug) {}

void LlamaFFN::forward(ModelContext &ctx) {
    if (w_ == nullptr) {
        throw std::invalid_argument("LlamaFFN requires weights");
    }

    auto &ws = ctx.workspace;
    const auto &w = *w_;

    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, ctx.meta.epsilon);
    ops::linear(ws.gate, ws.m_norm, w.gate_w, nullptr);
    ops::linear(ws.up, ws.m_norm, w.up_w, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    ops::linear(ws.down, ws.swiglu, w.down_w, nullptr);

    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

void run_llama_layer_stack(ModelContext &ctx, bool is_prefill) {
    if (ctx.attns.size() != ctx.meta.nlayer || ctx.ffns.size() != ctx.meta.nlayer) {
        throw std::invalid_argument("Llama layer weights do not match the model layer count");
    }
    if (ctx.runtime.cut_idx.size() < 2) {
        throw std::invalid_argument("Llama layer stack requires at least one sequence");
    }

    ctx.runtime.is_prefill = is_prefill;
    LlamaEmbedding{}.forward(ctx);
    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        LlamaAttention(&ctx.attns[layer], layer).forward(ctx);
        LlamaFFN(&ctx.ffns[layer], layer).forward(ctx);
    }

    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);
}

} // namespace llaisys::framework
