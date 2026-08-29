#include "logit.hpp"

#include "../../../../ops/embedding/op.hpp"
#include "../../../../ops/linear/op.hpp"
#include "../../../../ops/linear_w8a8/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/rms_norm_quant/op.hpp"
#include "../../weight/qwen2_w8_weights.hpp"

namespace llaisys::model {

Qwen2Logit::Qwen2Logit(Qwen2Weights &weights) : _w(weights) {}

void Qwen2Logit::forward(ModelContext &ctx) {
    forward(static_cast<Qwen2Context &>(ctx));
}

void Qwen2Logit::forward(Qwen2Context &ctx) {
    auto &ws = ctx.current();
    const auto &meta = ctx.meta;
    // 每序列最后一个 token 的 hidden state cut_idx 长度 = batch + 1
    const size_t batch_size = ws.cut_idx->shape()[0] - 1;
    auto last_token_indices = ws.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ws.x_part, last_token_indices, ws.x, -1);
    // lm_head 量化 (tie=False 时 _out_embed_i8 存在): rms_norm_quant + linear_w8a8 (dp4a gemv)
    // 467MB FP 表 → 233MB I8，decode 下带宽减半
    const auto *w8 = dynamic_cast<const Qwen2W8Weights *>(&this->_w);
    if (w8 && w8->_out_embed_i8) {
        ops::rms_norm_quant(ws.x_part_norm_q, ws.x_part_norm_a_scale, ws.x_part, this->_w._out_norm_w, meta.epsilon);
        ops::linear_w8a8(ws.logits, ws.x_part_norm_q, ws.x_part_norm_a_scale, w8->_out_embed_i8,
                         w8->_out_embed_scale, nullptr);
    } else {
        ops::rms_norm(ws.x_part_norm, ws.x_part, this->_w._out_norm_w, meta.epsilon);
        ops::linear(ws.logits, ws.x_part_norm, this->_w._out_embed, nullptr);
    }
}

} // namespace llaisys::model
