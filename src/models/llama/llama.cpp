#include "llama.hpp"

#include "layer/llama.hpp"
#include "meta/llama_meta.hpp"
#include "weight/llama_weights.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys {

Llama::Llama(llaisysDeviceType_t device, int *device_ids, int ndevice)
    : Qwen2(device, device_ids, ndevice, std::make_unique<model::LlamaContext>()) {}

/**
 * 设置 meta
 * 1. 检查 meta_ready
 * 2. 设置 meta
 * 3. 分配上下文内存
 */
void Llama::set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    if (this->_meta_ready) {
        throw std::runtime_error("set_meta called twice");
    }
    // 设置 meta
    this->_ctx->meta = model::LlamaMeta::from_map(kv);
    auto &meta = model::llama_meta(*this->_ctx);
    auto &lctx = this->llama_ctx();
    auto host_inv = meta.make_inv_freq();
    lctx.inv_freq = Tensor::create({meta.dh / 2}, LLAISYS_DTYPE_F32, this->_device_type, this->_device_ids[0]);
    lctx.inv_freq->load(host_inv.data());

    // 分配上下文内存
    this->allocate_runtime();
}

/**
 * 获取 llama 上下文
 * 1. 返回 llama 上下文
 */
model::LlamaContext &Llama::llama_ctx() {
    return static_cast<model::LlamaContext &>(*this->_ctx);
}

/**
 * 前向推理
 * 1. 绑定 kv cache
 * 2. 绑定上下文运行时
 * 3. 设置上下文运行时
 * 4. 前向推理 embedding
 * 5. 前向推理 attention
 * 6. 前向推理 ffn
 */
std::vector<int64_t> Llama::forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    // 获取上下文
    auto &ctx = this->llama_ctx();
    // 获取模型参数
    const auto &meta = model::llama_meta(ctx);
    // 设置上下文运行时
    ctx.runtime.is_prefill = is_prefill;

    // 绑定上下文运行时信息
    this->bind_context_runtime(pack, block_ids, is_prefill);

    // 绑定 kv cache
    this->bind_kv_caches();

    // 前向推理 embedding
    model::LlamaEmbedding{}.forward(ctx);

    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        // 前向推理 attention
        model::LlamaAttention(model::LlamaWeights::attn(ctx, layer), layer).forward(ctx);
        // 前向推理 ffn
        model::LlamaFFN(model::LlamaWeights::ffn(ctx, layer)).forward(ctx);
    }

    // 获取最后一层的 logits
    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    // 前向推理 embedding
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);
    // 前向推理 rms norm
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, meta.epsilon);
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);

    return this->sample_from_logits(ctx.workspace.logits, pack);
}

} // namespace llaisys
