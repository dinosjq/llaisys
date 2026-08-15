#include "stack.hpp"

#include "attention.hpp"
#include "embedding.hpp"
#include "ffn.hpp"
#include "../../../ops/embedding/op.hpp"
#include "../../../ops/linear/op.hpp"
#include "../../../ops/rms_norm/op.hpp"

#include <stdexcept>

namespace llaisys::framework {

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
        if (layer == 0) {
            ctx.workspace.capture_post_attn0 = ctx.workspace.x;
        }
        Qwen2FFN(&ctx.ffns[layer], layer).forward(ctx);
        if (layer == 0) {
            ctx.workspace.capture_post_ffn0 = ctx.workspace.x;
        }
    }

    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);
}

} // namespace llaisys::framework
