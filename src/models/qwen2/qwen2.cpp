#include "qwen2.hpp"

#include "layer/attn/attn.hpp"
#include "layer/ffn/ffn.hpp"
#include "layer/embedding/embedding.hpp"
#include "layer/logit/logit.hpp"
#include "weight/qwen2_weights.hpp"
#include "weight/qwen2_w8_weights.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/top_k/op.hpp"
#include "../../config.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace llaisys::model {

Qwen2::Qwen2(llaisysDeviceType_t device_type, int device_id) {
    this->_device_type = device_type;
    this->_device_id = device_id;
    this->_ctx = std::make_unique<Qwen2Context>();
}

Qwen2::~Qwen2() = default;

void Qwen2::set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    if (this->_meta_ready) {
        throw std::runtime_error("set_meta called twice");
    }
    this->_meta = std::make_unique<Qwen2Meta>(Qwen2Meta::cast(kv));
    this->_meta_ready = true;
    this->_allocate_runtime();
    // 权重槽位就绪后预构建层对象（引用固定，后续 forward 复用）
    this->_build_layers();
}

void Qwen2::set_weight(const std::string &hf_name, tensor_t tensor) {
    if (!this->_meta_ready) {
        throw std::runtime_error("set_weight before set_meta");
    }
    if (this->meta().quantize_weights) {
        if (this->_device_type != LLAISYS_DEVICE_NVIDIA) {
            throw std::runtime_error("quantize_weights requires NVIDIA device");
        }
        this->w8weights().set(hf_name, std::move(tensor));
    } else {
        this->weights().set(hf_name, std::move(tensor));
    }
    this->_weight_ready = true;
}

void Qwen2::_allocate_runtime() {
    const auto &meta = this->meta();
    const size_t nlayer = meta.nlayer;
    // 权重
    if (meta.quantize_weights) {
        if (this->_device_type != LLAISYS_DEVICE_NVIDIA) {
            throw std::runtime_error("quantize_weights requires NVIDIA device");
        }
        this->_weights = std::make_unique<Qwen2W8Weights>();
        this->w8weights().init(nlayer);
    } else {
        this->_weights = std::make_unique<Qwen2Weights>();
        this->weights().init(nlayer);
    }
    // 中间张量 + KV cache
    this->ctx().allocate(this->_device_type, this->_device_id, this->meta());
}

/**
 * 组装全部层：embedding → (attn + ffn) × nlayer → logit
 * set_meta 后一次性构建，forward 复用
 */
void Qwen2::_build_layers() {
    this->_clear_layers();
    auto &w = this->weights();
    this->_add_layer(std::make_unique<Qwen2Embedding>(w));
    for (size_t layer = 0; layer < this->meta().nlayer; ++layer) {
        this->_add_layer(std::make_unique<Qwen2Attention>(w.attn(layer), layer));
        this->_add_layer(std::make_unique<Qwen2FFN>(w.ffn(layer)));
    }
    this->_add_layer(std::make_unique<Qwen2Logit>(w));
}

engine::BatchOutput Qwen2::forward(const engine::BatchInput &input) {
    // 获取上下文 / 元信息 / 批次输入
    auto &ctx = this->ctx();
    const auto &meta = this->meta();
    const auto &pack = std::get<engine::BatchPack>(input);

    // 绑定上下文运行时信息
    ctx.bind(input, meta);

    // 遍历预构建的层
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

/** 测试用：获取最后一层的 logits
 * 1. 获取词汇表大小
 * 2. 获取最后一层的 logits
 */
tensor_t Qwen2::last_logits(size_t batch_size) const {
    const auto &ctx = this->ctx();
    return ctx.current().logits;
}

} // namespace llaisys::model
