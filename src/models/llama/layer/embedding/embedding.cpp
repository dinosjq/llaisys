#include "embedding.hpp"

#include "../../../../ops/embedding/op.hpp"

namespace llaisys::model {

void LlamaEmbedding::forward(ModelContext &ctx) {
    forward(static_cast<LlamaContext &>(ctx));
}

void LlamaEmbedding::forward(LlamaContext &ctx) {
    auto &ws = ctx.workspace;
    auto &runtime = ctx.runtime;

    auto token_ids = Tensor::create({runtime.token_ids.size()}, LLAISYS_DTYPE_I64, ws.x->deviceType(), ws.x->deviceId());
    token_ids->load(runtime.token_ids.data());
    ops::embedding(ws.x, token_ids, ctx.in_embed);
}

} // namespace llaisys::model
