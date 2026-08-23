#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../tensor/tensor.hpp"
#include "../../model/model.hpp"
#include "../../model/layer/layer.hpp"
#include "../../sampler/sampler.hpp"
#include "runtime/qwen2_context.hpp"
#include "meta/qwen2_meta.hpp"
#include "weight/qwen2_weights.hpp"
#include "weight/qwen2_w8_weights.hpp"

namespace llaisys::model {

class Qwen2;
using Qwen2_t = std::unique_ptr<Qwen2>;

class Qwen2 : public model::Model {
protected:
    void _allocate_runtime();
    // 内部层组装
    void _build_layers();
    void _clear_layers() { this->_layers.clear(); }
    void _add_layer(std::unique_ptr<Layer> layer) { this->_layers.push_back(std::move(layer)); }
    llaisys::Sampler _sampler;

public:
    Qwen2(llaisysDeviceType_t device_type, int device_id);
    ~Qwen2() override;

    // load
    void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) override;
    void set_weight(const std::string &hf_name, tensor_t tensor) override;

    // forward
    engine::BatchOutput forward(const engine::BatchInput &input) override;

    int64_t end_token() const override { return this->meta().end_token; }

    // 测试用：获取最后一层的 logits
    tensor_t last_logits(size_t batch_size) const;

    Qwen2Meta &meta() { return dynamic_cast<Qwen2Meta &>(*this->_meta); };
    const Qwen2Meta &meta() const { return dynamic_cast<const Qwen2Meta &>(*this->_meta); };

    Qwen2Context &ctx() { return dynamic_cast<Qwen2Context &>(*this->_ctx); };
    const Qwen2Context &ctx() const { return dynamic_cast<const Qwen2Context &>(*this->_ctx); };

    Qwen2Weights &weights() { return dynamic_cast<Qwen2Weights &>(*this->_weights); };
    const Qwen2Weights &weights() const { return dynamic_cast<const Qwen2Weights &>(*this->_weights); };

    Qwen2W8Weights &w8weights() { return dynamic_cast<Qwen2W8Weights &>(*this->_weights); };
    const Qwen2W8Weights &w8weights() const { return dynamic_cast<const Qwen2W8Weights &>(*this->_weights); };
    
};
} // namespace llaisys
