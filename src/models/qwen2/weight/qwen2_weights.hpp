#pragma once

#include "../../../model/weight/attn_weights.hpp"
#include "../../../model/weight/ffn_weights.hpp"
#include "../../../model/runtime/model_context.hpp"
#include "../meta/qwen2_meta.hpp"
#include "../../../tensor/tensor.hpp"

#include <memory>
#include <string>
#include <utility>

namespace llaisys::model {

class QwenAttnWeights : public AttnWeights {
public:
    tensor_t norm_w, q_w, q_b, k_w, k_b, v_w, v_b, o_w;
};
using qwen_attn_weights_t = std::shared_ptr<QwenAttnWeights>;

class QwenFfnWeights : public FfnWeights {
public:
    tensor_t norm_w, gate_w, up_w, down_w;
};
using qwen_ffn_weights_t = std::shared_ptr<QwenFfnWeights>;

class Qwen2Weights {
public:
    static qwen_attn_weights_t attn(ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenAttnWeights>(ctx.attns.at(layer));
    }

    static qwen_attn_weights_t attn(const ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenAttnWeights>(ctx.attns.at(layer));
    }

    static qwen_ffn_weights_t ffn(ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenFfnWeights>(ctx.ffns.at(layer));
    }

    static qwen_ffn_weights_t ffn(const ModelContext &ctx, size_t layer) {
        return std::static_pointer_cast<QwenFfnWeights>(ctx.ffns.at(layer));
    }

    static void init(ModelContext &ctx) {
        const size_t nlayer = qwen_meta(ctx).nlayer;
        ctx.resize_layers(nlayer);
        for (size_t i = 0; i < nlayer; ++i) {
            ctx.attns[i] = std::make_shared<QwenAttnWeights>();
            ctx.ffns[i] = std::make_shared<QwenFfnWeights>();
        }
    }

    // Returns true if the HF name was recognized and applied.
    static bool set_weight(ModelContext &ctx, const std::string &hf_name, tensor_t tensor) {
        if (hf_name == "model.embed_tokens.weight") {
            ctx.in_embed = std::move(tensor);
            return true;
        }
        if (hf_name == "lm_head.weight") {
            ctx.out_embed = std::move(tensor);
            return true;
        }
        if (hf_name == "model.norm.weight") {
            ctx.out_norm_w = std::move(tensor);
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
        if (layer >= ctx.attns.size() || layer >= ctx.ffns.size() || !ctx.attns[layer] || !ctx.ffns[layer]) {
            return false;
        }
        const std::string suffix = hf_name.substr(dot + 1);
        auto a = attn(ctx, layer);
        auto f = ffn(ctx, layer);

        if (suffix == "input_layernorm.weight") {
            a->norm_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.q_proj.weight") {
            a->q_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.q_proj.bias") {
            a->q_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.k_proj.weight") {
            a->k_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.k_proj.bias") {
            a->k_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.v_proj.weight") {
            a->v_w = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.v_proj.bias") {
            a->v_b = std::move(tensor);
            return true;
        }
        if (suffix == "self_attn.o_proj.weight") {
            a->o_w = std::move(tensor);
            return true;
        }
        if (suffix == "post_attention_layernorm.weight") {
            f->norm_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.gate_proj.weight") {
            f->gate_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.up_proj.weight") {
            f->up_w = std::move(tensor);
            return true;
        }
        if (suffix == "mlp.down_proj.weight") {
            f->down_w = std::move(tensor);
            return true;
        }
        return false;
    }
};

} // namespace llaisys::model
