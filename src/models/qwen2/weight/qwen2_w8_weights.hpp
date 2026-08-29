#pragma once

#include "../../../model/weight/utils/quantize_per_channel.hpp"
#include "../../../utils/check.hpp"
#include "qwen2_weights.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

// W8A16 量化层：在纯 FP 权重基础上补充 per-channel scale
struct QwenW8AttnWeights : QwenAttnWeights {
    tensor_t q_scale, k_scale, v_scale, o_scale;
};

struct QwenW8FfnWeights : QwenFfnWeights {
    tensor_t gate_scale, up_scale, down_scale;
};

class Qwen2W8Weights : public Qwen2Weights {
public:
    // lm_head 量化权重（仅 tie=False 时存在；tie=True 时 lm_head 复用 embedding，不量化）
    tensor_t _out_embed_i8;    // [vocab, hs] I8 + Hadamard 旋转
    tensor_t _out_embed_scale; // per-channel F32 scale

    // W8 路径：所有层都创建为量化层类型（基类 set 处理 norm/bias，本类处理量化 weight）
    void init(const size_t nlayer) override {
        this->_nlayer = nlayer;
        this->_attns.resize(nlayer);
        this->_ffns.resize(nlayer);
        for (size_t i = 0; i < nlayer; ++i) {
            this->_attns[i] = std::make_unique<QwenW8AttnWeights>();
            this->_ffns[i] = std::make_unique<QwenW8FfnWeights>();
        }
    }

    bool set(const std::string &hf_name, tensor_t tensor) {
        if (!tensor) {
            return false;
        }
        auto quant_assign = [](tensor_t &w_slot, tensor_t &scale_slot, tensor_t src) {
            auto [w_i8, scale] = weight::utils::quantize_weight_per_channel(src);
            w_slot = std::move(w_i8);
            scale_slot = std::move(scale);
        };
        static const char kPrefix[] = "model.layers.";
        if (hf_name.rfind(kPrefix, 0) == 0) {
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
            // init() 保证该层是量化类型；dynamic_cast 失败则不是 W8 路径
            auto *attn = dynamic_cast<QwenW8AttnWeights *>(this->_attns[layer].get());
            auto *ffn = dynamic_cast<QwenW8FfnWeights *>(this->_ffns[layer].get());
            ASSERT(attn != nullptr && ffn != nullptr, "Qwen2W8Weights: layer is not quantized type");

            if (suffix == "self_attn.q_proj.weight") {
                quant_assign(attn->q_w, attn->q_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.k_proj.weight") {
                quant_assign(attn->k_w, attn->k_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.v_proj.weight") {
                quant_assign(attn->v_w, attn->v_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.o_proj.weight") {
                quant_assign(attn->o_w, attn->o_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.gate_proj.weight") {
                quant_assign(ffn->gate_w, ffn->gate_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.up_proj.weight") {
                quant_assign(ffn->up_w, ffn->up_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.down_proj.weight") {
                quant_assign(ffn->down_w, ffn->down_scale, std::move(tensor));
                return true;
            }
        }

        // lm_head 独立权重（tie=False 时量化；tie=True 时不在 state dict 中，_out_embed 复用 embedding）
        if (hf_name == "lm_head.weight") {
            auto [w_i8, scale] = weight::utils::quantize_weight_per_channel(std::move(tensor));
            this->_out_embed_i8 = std::move(w_i8);
            this->_out_embed_scale = std::move(scale);
            return true;
        }

        return Qwen2Weights::set(hf_name, std::move(tensor));
    }
};

} // namespace llaisys::model
