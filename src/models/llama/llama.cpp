#include "llama.hpp"

#include "layer/attn/attn.hpp"
#include "layer/ffn/ffn.hpp"
#include "layer/embedding/embedding.hpp"
#include "../qwen2/layer/logit/logit.hpp"
#include "meta/llama_meta.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

Llama::Llama(llaisysDeviceType_t device, int device_id)
    : Qwen2(device, device_id) {
    // Llama 需要扩展上下文（预计算 inv_freq）
    this->_ctx = std::make_unique<LlamaContext>();
}

/**
 * 设置 meta
 * 1. 检查 meta_ready
 * 2. 设置 meta（Qwen2 布局 + Llama 专属默认）
 * 3. 分配上下文内存 / 权重槽位
 * 4. 预计算 RoPE inv_freq
 */
void Llama::set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    if (this->_meta_ready) {
        throw std::runtime_error("set_meta called twice");
    }
    this->_meta = LlamaMeta::from_map(kv);
    this->_meta_ready = true;
    // 权重槽位 + 中间张量 + KV cache
    this->_allocate_runtime();
    // 预构建层对象（Llama 组装）
    this->_build_layers();
    // 预计算 inv_freq[dh/2]
    auto &meta = dynamic_cast<LlamaMeta &>(*this->_meta);
    auto host_inv = meta.make_inv_freq();
    auto &lctx = this->llama_ctx();
    lctx.inv_freq = Tensor::create({meta.dh / 2}, LLAISYS_DTYPE_F32, this->_device_type, this->_device_id);
    lctx.inv_freq->load(host_inv.data());
}

/**
 * 获取 llama 上下文
 * 1. 返回 llama 上下文
 */
LlamaContext &Llama::llama_ctx() {
    return static_cast<LlamaContext &>(*this->_ctx);
}

/**
 * 组装全部层：embedding → (attn + ffn) × nlayer → logit
 * Llama 用独立的 Attention/FFN（rope 走 inv_freq），logit 复用 Qwen2 实现
 */
void Llama::_build_layers() {
    this->_clear_layers();
    auto &w = this->weights();
    this->_add_layer(std::make_unique<LlamaEmbedding>(w));
    for (size_t layer = 0; layer < this->meta().nlayer; ++layer) {
        this->_add_layer(std::make_unique<LlamaAttention>(w.attn(layer), layer));
        this->_add_layer(std::make_unique<LlamaFFN>(w.ffn(layer)));
    }
    this->_add_layer(std::make_unique<Qwen2Logit>(w));
}

/**
 * 前向推理
 * 1. 绑定上下文运行时
 * 2. 遍历预构建的层（set_meta 时已构建）
 * 3. 采样
 */
engine::BatchOutput Llama::forward(const engine::BatchInput &input) {
    auto &ctx = this->llama_ctx();
    const auto &meta = dynamic_cast<const LlamaMeta &>(this->meta());
    const auto &pack = std::get<engine::BatchPack>(input);

    // 绑定上下文运行时信息
    ctx.bind(input, meta);

    // 遍历预构建的层（防御：未构建则构建）
    if (this->_layers.empty()) {
        this->_build_layers();
    }
    for (auto &layer : this->_layers) {
        layer->forward(ctx);
    }

    // 采样 token 独立 Sampler
    auto token_ids = this->_sampler.sample(ctx.current().logits, pack, this->_device_type, this->_device_id, meta.dtype);
    engine::BatchOutput output;
    output.emplace<std::vector<int64_t>>(std::move(token_ids));
    return output;
}

} // namespace llaisys::model
