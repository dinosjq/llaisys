#include "embedding.hpp"

#include "../../../ops/embedding/op.hpp"

namespace llaisys::framework {

void Qwen2Embedding::forward(ModelContext &ctx) {
    auto token_ids = Tensor::create({ctx.runtime.token_ids.size()}, LLAISYS_DTYPE_I64, ctx.workspace.x->deviceType(), ctx.workspace.x->deviceId());
    token_ids->load(ctx.runtime.token_ids.data());
    ops::embedding(ctx.workspace.x, token_ids, ctx.in_embed);
}

} // namespace llaisys::framework
