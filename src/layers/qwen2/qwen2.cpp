#include "qwen2.hpp"

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

void Qwen2Embedding::forward(ModelContext &ctx) {
    auto &ws = ctx.workspace;
    auto &runtime = ctx.runtime;

    // embedding: 根据 token ids 从 embedding 权重矩阵中提取对应的语义向量
    auto token_ids = Tensor::create({runtime.token_ids.size()}, LLAISYS_DTYPE_I64, ws.x->deviceType(), ws.x->deviceId());
    token_ids->load(runtime.token_ids.data());
    ops::embedding(ws.x, token_ids, ctx.in_embed);
}

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

Qwen2FFN::Qwen2FFN(const FfnWeights *weights, size_t layer_index_for_debug) : w_(weights), layer_index_for_debug_(layer_index_for_debug) {}

void Qwen2FFN::forward(ModelContext &ctx) {
    if (w_ == nullptr) {
        throw std::invalid_argument("Qwen2FFN requires weights");
    }

    auto &ws = ctx.workspace;
    const auto &w = *w_;

    // 多层感知机 MLP block
    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, ctx.meta.epsilon);
    ops::linear(ws.gate, ws.m_norm, w.gate_w, nullptr);
    ops::linear(ws.up, ws.m_norm, w.up_w, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    ops::linear(ws.down, ws.swiglu, w.down_w, nullptr);

    // 残差连接: x + MLP(x)
    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

namespace {

// Stable capture: copy to CPU so later workspace reuse cannot mutate the snapshot.
tensor_t stable_capture(const tensor_t &tensor) {
    return tensor->to(LLAISYS_DEVICE_CPU, 0);
}

} // namespace

void run_qwen2_layer_stack(ModelContext &ctx, bool is_prefill) {
    if (ctx.attns.size() != ctx.meta.nlayer || ctx.ffns.size() != ctx.meta.nlayer) {
        throw std::invalid_argument("Qwen2 layer weights do not match the model layer count");
    }
    if (ctx.runtime.cut_idx.size() < 2) {
        throw std::invalid_argument("Qwen2 layer stack requires at least one sequence");
    }

    ctx.runtime.is_prefill = is_prefill;
    Qwen2Embedding{}.forward(ctx);
    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        Qwen2Attention(&ctx.attns[layer], layer).forward(ctx);
        if (layer == 0 && ctx.enable_layer0_capture) {
            // Capture after attention residual swap into current x.
            ctx.workspace.capture_post_attn0 = stable_capture(ctx.workspace.x);
        }
        Qwen2FFN(&ctx.ffns[layer], layer).forward(ctx);
        if (layer == 0 && ctx.enable_layer0_capture) {
            // Capture after FFN residual swap into current x.
            ctx.workspace.capture_post_ffn0 = stable_capture(ctx.workspace.x);
        }
    }

    // 先去提取 last-token 隐藏状态
    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);

    // RMS-norm 均方根归一化
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);

    // 把隐藏向量映射到词表维度生成 logits
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);
}

} // namespace llaisys::framework
