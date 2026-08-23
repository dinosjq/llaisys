#include "embedding.hpp"

#include "../../../../ops/embedding/op.hpp"

namespace llaisys::model {

Qwen2Embedding::Qwen2Embedding(Qwen2Weights &weights) : _w(weights) {}

void Qwen2Embedding::forward(ModelContext &ctx) {
    forward(static_cast<Qwen2Context &>(ctx));
}

void Qwen2Embedding::forward(Qwen2Context &ctx) {
    auto &ws = ctx.current();
    ops::embedding(ws.x, ws.token_ids, this->_w._in_embed);
}

} // namespace llaisys::model
