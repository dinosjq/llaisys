#pragma once

#include "qwen2_weights.hpp"
#include "../../../model/weight/utils/quantize_per_channel.hpp"

#include <stdexcept>
#include <utility>

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

    // Online-quantize projection weights; other tensors stay FP via Qwen2Weights routing.
    static bool set_weight(ModelContext &ctx, const std::string &hf_name, tensor_t tensor) {
        if (!tensor) {
            return false;
        }
        auto quant_assign = [&](tensor_t &w_slot, tensor_t &scale_slot, tensor_t src) {
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
            if (layer >= ctx.attns.size() || layer >= ctx.ffns.size() || !ctx.attns[layer] || !ctx.ffns[layer]) {
                return false;
            }
            const std::string suffix = hf_name.substr(dot + 1);
            auto a = attn(ctx, layer);
            auto f = ffn(ctx, layer);

            if (suffix == "self_attn.q_proj.weight") {
                quant_assign(a->q_w, a->q_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.k_proj.weight") {
                quant_assign(a->k_w, a->k_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.v_proj.weight") {
                quant_assign(a->v_w, a->v_scale, std::move(tensor));
                return true;
            }
            if (suffix == "self_attn.o_proj.weight") {
                quant_assign(a->o_w, a->o_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.gate_proj.weight") {
                quant_assign(f->gate_w, f->gate_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.up_proj.weight") {
                quant_assign(f->up_w, f->up_scale, std::move(tensor));
                return true;
            }
            if (suffix == "mlp.down_proj.weight") {
                quant_assign(f->down_w, f->down_scale, std::move(tensor));
                return true;
            }
        }

        return Qwen2Weights::set_weight(ctx, hf_name, std::move(tensor));
    }
};

} // namespace llaisys::model
