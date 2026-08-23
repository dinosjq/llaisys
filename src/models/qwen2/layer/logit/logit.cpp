#include "logit.hpp"

#include "../../../../ops/embedding/op.hpp"
#include "../../../../ops/linear/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"

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
    ops::rms_norm(ws.x_part_norm, ws.x_part, this->_w._out_norm_w, meta.epsilon);
    ops::linear(ws.logits, ws.x_part_norm, this->_w._out_embed, nullptr);
}

} // namespace llaisys::model
