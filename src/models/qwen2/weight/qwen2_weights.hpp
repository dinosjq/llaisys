#pragma once

#include "../../../model/weight/model_weights.hpp"
#include "../../../tensor/tensor.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace llaisys::model {

// 单层 attention 权重（纯 FP；W8 量化由 QwenW8AttnWeights 补充 scale）
struct QwenAttnWeights {
    virtual ~QwenAttnWeights() = default;
    tensor_t norm_w;
    tensor_t q_w, q_b, k_w, k_b, v_w, v_b, o_w;
};

// 单层 ffn 权重（纯 FP；W8 量化由 QwenW8FfnWeights 补充 scale）
struct QwenFfnWeights {
    virtual ~QwenFfnWeights() = default;
    tensor_t norm_w;
    tensor_t gate_w, up_w, down_w;
};

class Qwen2Weights : public ModelWeights {
public:
    Qwen2Weights() = default;
    ~Qwen2Weights() override = default;

    Qwen2Weights(const Qwen2Weights &) = delete;
    Qwen2Weights &operator=(const Qwen2Weights &) = delete;
    Qwen2Weights(Qwen2Weights &&) = delete;
    Qwen2Weights &operator=(Qwen2Weights &&) = delete;

    size_t _nlayer = 0;
    // in / out
    tensor_t _in_embed;
    tensor_t _out_embed;
    tensor_t _out_norm_w;
    // 分层权重（多态存储：W8 时指向 QwenW8* 派生类型）
    std::vector<std::unique_ptr<QwenAttnWeights>> _attns;
    std::vector<std::unique_ptr<QwenFfnWeights>> _ffns;

    virtual void init(const size_t nlayer) {
        this->_nlayer = nlayer;
        this->_attns.resize(nlayer);
        this->_ffns.resize(nlayer);
        for (size_t i = 0; i < nlayer; ++i) {
            this->_attns[i] = std::make_unique<QwenAttnWeights>();
            this->_ffns[i] = std::make_unique<QwenFfnWeights>();
        }
    }

    QwenAttnWeights &attn(size_t layer) { return *this->_attns[layer]; }
    const QwenAttnWeights &attn(size_t layer) const { return *this->_attns[layer]; }
    QwenFfnWeights &ffn(size_t layer) { return *this->_ffns[layer]; }
    const QwenFfnWeights &ffn(size_t layer) const { return *this->_ffns[layer]; }

    // set weight（按 HF 名称解析到对应层对象）
    bool set(const std::string &hf_name, tensor_t tensor) {
        if (hf_name == "model.embed_tokens.weight") {
            this->_in_embed = std::move(tensor);
            return true;
        }
        if (hf_name == "lm_head.weight") {
            this->_out_embed = std::move(tensor);
            return true;
        }
        if (hf_name == "model.norm.weight") {
            this->_out_norm_w = std::move(tensor);
            return true;
        }
        // model.layers.{i}....
        static const char kPrefix[] = "model.layers.";
        if (hf_name.rfind(kPrefix, 0) != 0) {
            return false;
        }
        const size_t start = sizeof(kPrefix) - 1;
        const size_t dot = hf_name.find('.', start);
        if (dot == std::string::npos) {
            return false;
        }
        size_t layer = 0;
        try {
            layer = static_cast<size_t>(std::stoul(hf_name.substr(start, dot - start)));
        } catch (...) {
            return false;
        }
        if (layer >= this->_nlayer) {
            return false;
        }
        const std::string suffix = hf_name.substr(dot + 1);
        auto &attn = this->attn(layer);
        auto &ffn = this->ffn(layer);
        if (suffix == "input_layernorm.weight") {
            attn.norm_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.q_proj.weight") {
            attn.q_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.q_proj.bias") {
            attn.q_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.k_proj.weight") {
            attn.k_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.k_proj.bias") {
            attn.k_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.v_proj.weight") {
            attn.v_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.v_proj.bias") {
            attn.v_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.o_proj.weight") {
            attn.o_w = std::move(tensor);
            return true;
        }
        if (suffix == "post_attention_layernorm.weight") {
            ffn.norm_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.gate_proj.weight") {
            ffn.gate_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.up_proj.weight") {
            ffn.up_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.down_proj.weight") {
            ffn.down_w = std::move(tensor);
            return true;
        }
        return false;
    }
};

using qwen_weights_t = std::unique_ptr<Qwen2Weights>;

} // namespace llaisys::model
