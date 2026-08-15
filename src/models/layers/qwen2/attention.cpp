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

    auto &ws = ctx.workspace;
    const auto &meta = ctx.meta;
    const auto &w = *w_;

    const size_t token_count = ws.x->shape()[0];
    // 计算 self_attention 所需的参数 scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(meta.dh));

    // RMS-norm 均方根归一化
    ops::rms_norm(ws.x_norm, ws.x, w.norm_w, meta.epsilon);

    // 初始化 Q K V
    ops::linear(ws.q, ws.x_norm, w.q_w, w.q_b);
    ops::linear(ws.k, ws.x_norm, w.k_w, w.k_b);
    ops::linear(ws.v, ws.x_norm, w.v_w, w.v_b);

    // 旋转位置编码 rope: 将位置信息融入进 Q / K 当中
    ops::rope(ws.q_rope, ws.q->view({token_count, meta.nh, meta.dh}), ws.pos_ids, meta.theta);
    ops::rope(ws.k_rope, ws.k->view({token_count, meta.nkvh, meta.dh}), ws.pos_ids, meta.theta);

    // 按块加载到 kv cache
    ops::kv_cache_move(ctx.k_caches[layer_], ws.k_rope, ws.block_ids, ws.cut_idx, ws.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer_], ws.v->view({token_count, meta.nkvh, meta.dh}), ws.block_ids, ws.cut_idx, ws.pos_ids,
                       ctx.runtime.max_seq_len);

    // 自注意力: paged_attention(flash v2 / flash decoding)
    // tot_block_num is batch block-table cumulative count, not k_cache pool capacity.
    ops::paged_attention(ws.attn_val, ws.q_rope, ctx.k_caches[layer_], ctx.v_caches[layer_], ws.block_ids, ws.cut_idx, ws.tot_len,
                         ctx.runtime.max_seq_len, ctx.runtime.tot_block_num, scale, ctx.runtime.is_prefill, ws.attn_acc, ws.attn_sum,
                         ws.attn_max);

    // 得到多头注意力输出投影
    ops::linear(ws.attn_out, ws.attn_val->view({token_count, meta.hs}), w.o_w, nullptr);

    // 实现残差连接: x(上一层信息) + attn_out(当前层信息)
    ops::add(ws.x_attn, ws.x, ws.attn_out);
    std::swap(ws.x, ws.x_attn);
}

} // namespace llaisys::framework
