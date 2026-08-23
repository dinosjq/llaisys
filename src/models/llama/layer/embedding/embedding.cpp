#include "embedding.hpp"

#include "../../../../ops/embedding/op.hpp"

namespace llaisys::model {

LlamaEmbedding::LlamaEmbedding(Qwen2Weights &weights) : _w(weights) {}

void LlamaEmbedding::forward(ModelContext &ctx) {
    forward(static_cast<LlamaContext &>(ctx));
}

void LlamaEmbedding::forward(LlamaContext &ctx) {
    auto &ws = ctx.current();
    ops::embedding(ws.x, ws.token_ids, this->_w._in_embed);
}

} // namespace llaisys::model
