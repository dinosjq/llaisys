#include "llama.hpp"

#include "../../layer/llama/llama.hpp"
#include "../../layer/utils/rope_inv_freq.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"

namespace llaisys {

namespace {

// Llama-3 rope_scaling 中几乎不变的部分（变的是 meta.rope_scale）
constexpr float kLlamaRopeLowFreqFactor = 1.f;
constexpr float kLlamaRopeHighFreqFactor = 4.f;
constexpr int64_t kLlamaRopeOriginalMaxPos = 8192;

} // namespace

Llama::Llama(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice)
    : Qwen2(meta, device, device_ids, ndevice, std::make_unique<model::LlamaContext>()) {
    auto &lctx = this->llama_ctx();
    auto host_inv = layers::utils::make_rope_inv_freq(this->_meta.dh, this->_meta.theta);
    if (this->_meta.rope_scale > 1.f) {
        host_inv = layers::utils::apply_llama3_rope_scaling(std::move(host_inv), this->_meta.rope_scale,
                                                            kLlamaRopeLowFreqFactor, kLlamaRopeHighFreqFactor,
                                                            kLlamaRopeOriginalMaxPos);
    }
    lctx.inv_freq = Tensor::create({this->_meta.dh / 2}, LLAISYS_DTYPE_F32, this->_device_type, this->_device_ids[0]);
    lctx.inv_freq->load(host_inv.data());
}

model::LlamaContext &Llama::llama_ctx() {
    return static_cast<model::LlamaContext &>(*this->_ctx);
}

std::vector<int64_t> Llama::forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) {
    this->bind_kv_caches();
    this->bind_context_runtime(pack, block_ids, is_prefill);

    auto &ctx = this->llama_ctx();
    ctx.runtime.is_prefill = is_prefill;

    model::LlamaEmbedding{}.forward(ctx);

    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        model::LlamaAttention(&ctx.attns[layer], layer).forward(ctx);
        model::LlamaFFN(&ctx.ffns[layer]).forward(ctx);
    }

    const size_t batch_size = ctx.runtime.cut_idx.size() - 1;
    auto last_token_indices = ctx.workspace.cut_idx->slice(0, 1, batch_size + 1);
    ops::embedding(ctx.workspace.x_part, last_token_indices, ctx.workspace.x, -1);
    ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);
    ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);

    return this->sample_from_logits(ctx.workspace.logits, pack);
}

} // namespace llaisys
