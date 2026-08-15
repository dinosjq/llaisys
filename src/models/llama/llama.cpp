#include "llama.hpp"

#include "../../layers/llama/llama.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"

namespace llaisys {

std::vector<int64_t> Llama::forward(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    this->bind_kv_caches();
    this->bind_context_runtime(pack, block_ids, is_prefill);

    core::ModelContext &ctx = this->_ctx;
    ctx.runtime.is_prefill = is_prefill;

    // 显式组装 Layer
    // embedding
    core::LlamaEmbedding{}.forward(ctx);

    // 逐层: Attention → FFN
    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        core::LlamaAttention(&ctx.attns[layer], layer).forward(ctx);
        core::LlamaFFN(&ctx.ffns[layer]).forward(ctx);
    }

    // 先去提取 last-token 隐藏状态
    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);

    // RMS-norm 均方根归一化
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);

    // 把隐藏向量映射到词表维度（voc）生成未归一化的打分（logits）
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);

    return this->sample_from_logits(ctx.workspace.logits, pack);
}

} // namespace llaisys
