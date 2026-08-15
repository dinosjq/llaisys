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

namespace llaisys::core {

void LlamaEmbedding::forward(ModelContext &ctx) {
    auto &ws = ctx.workspace;
    auto &runtime = ctx.runtime;

    // embedding: 根据 token ids 从 embedding 权重矩阵中提取对应的语义向量
    auto token_ids = Tensor::create({runtime.token_ids.size()}, LLAISYS_DTYPE_I64, ws.x->deviceType(), ws.x->deviceId());
    token_ids->load(runtime.token_ids.data());
    ops::embedding(ws.x, token_ids, ctx.in_embed);
}

LlamaAttention::LlamaAttention(const AttnWeights *weights, size_t layer)
    : _w(weights), _layer(layer) {}

void LlamaAttention::forward(ModelContext &ctx) {
    if (this->_w == nullptr || this->_layer >= ctx.k_caches.size() || this->_layer >= ctx.v_caches.size()) {
        throw std::invalid_argument("LlamaAttention requires weights and per-layer KV caches");
    }

    auto &ws = ctx.workspace;
    const auto &meta = ctx.meta;
    const auto &w = *this->_w;
    const size_t layer = this->_layer;

    const size_t nh = meta.nh;
    const size_t nkvh = meta.nkvh;
    const size_t dh = meta.dh;
    const size_t hs = meta.hs;
    const size_t token_count = ws.x->shape()[0];
    // 计算 self_attention 所需的参数 scale
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // RMS-norm 均方根归一化
    ops::rms_norm(ws.x_norm, ws.x, w.norm_w, meta.epsilon);

    // 初始化 Q K V（Llama: attention_bias=false，bias 传 nullptr）
    ops::linear(ws.q, ws.x_norm, w.q_w, nullptr);
    ops::linear(ws.k, ws.x_norm, w.k_w, nullptr);
    ops::linear(ws.v, ws.x_norm, w.v_w, nullptr);

    // 旋转位置编码 rope: 将位置信息融入进 Q / K 当中
    // 注意: Llama-3.2 的 rope_scaling(llama3) 尚未实现，当前仅用 meta.theta
    ops::rope(ws.q_rope, ws.q->view({token_count, nh, dh}), ws.pos_ids, meta.theta);
    ops::rope(ws.k_rope, ws.k->view({token_count, nkvh, dh}), ws.pos_ids, meta.theta);

    // 按块加载到 kv cache
    ops::kv_cache_move(ctx.k_caches[layer], ws.k_rope, ws.block_ids, ws.cut_idx, ws.pos_ids, ctx.runtime.max_seq_len);
    ops::kv_cache_move(ctx.v_caches[layer], ws.v->view({token_count, nkvh, dh}), ws.block_ids, ws.cut_idx, ws.pos_ids,
                       ctx.runtime.max_seq_len);

    // 自注意力: paged_attention(flash v2 / flash decoding)
    ops::paged_attention(ws.attn_val, ws.q_rope, ctx.k_caches[layer], ctx.v_caches[layer], ws.block_ids, ws.cut_idx,
                         ws.tot_len, ctx.runtime.max_seq_len, ctx.runtime.tot_block_num, scale, ctx.runtime.is_prefill,
                         ws.attn_acc, ws.attn_sum, ws.attn_max);

    // 得到多头注意力输出投影
    ops::linear(ws.attn_out, ws.attn_val->view({token_count, hs}), w.o_w, nullptr);

    // 实现残差连接: x(上一层信息) + attn_out(当前层信息)
    ops::add(ws.x_attn, ws.x, ws.attn_out);
    std::swap(ws.x, ws.x_attn);
}

LlamaFFN::LlamaFFN(const FfnWeights *weights) : _w(weights) {}

void LlamaFFN::forward(ModelContext &ctx) {
    if (this->_w == nullptr) {
        throw std::invalid_argument("LlamaFFN requires weights");
    }

    auto &ws = ctx.workspace;
    const auto &w = *this->_w;

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

} // namespace llaisys::core
