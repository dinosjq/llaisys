#pragma once

#include "qwen2_weights.hpp"

namespace llaisys::model {

class QwenW8AttnWeights : public QwenAttnWeights {
public:
    tensor_t q_scale, k_scale, v_scale, o_scale;
};
using qwen_w8_attn_weights_t = std::shared_ptr<QwenW8AttnWeights>;

class QwenW8FfnWeights : public QwenFfnWeights {
public:
    tensor_t gate_scale, up_scale, down_scale;
};
using qwen_w8_ffn_weights_t = std::shared_ptr<QwenW8FfnWeights>;

class Qwen2W8Weights {
public:
    static qwen_w8_attn_weights_t attn(ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenW8AttnWeights>(ctx.attns.at(layer));
    }

    static qwen_w8_ffn_weights_t ffn(ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenW8FfnWeights>(ctx.ffns.at(layer));
    }

    static void init(ModelContext &ctx) {
        const size_t nlayer = qwen_meta(ctx).nlayer;
        ctx.resize_layers(nlayer);
        for (size_t i = 0; i < nlayer; ++i) {
            ctx.attns[i] = std::make_shared<QwenW8AttnWeights>();
            ctx.ffns[i] = std::make_shared<QwenW8FfnWeights>();
        }
    }

    static bool set_weight(ModelContext &ctx, const std::string &hf_name, tensor_t tensor) {
        return Qwen2Weights::set_weight(ctx, hf_name, std::move(tensor));
    }
};

} // namespace llaisys::model
