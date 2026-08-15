#include "model.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::framework {

namespace {

void validate_attn_layer(const ModelContext &ctx, size_t layer) {
    if (layer >= ctx.meta.nlayer || layer >= ctx.attns.size()) {
        throw std::invalid_argument("attention weight layer is out of range");
    }
}

void validate_ffn_layer(const ModelContext &ctx, size_t layer) {
    if (layer >= ctx.meta.nlayer || layer >= ctx.ffns.size()) {
        throw std::invalid_argument("FFN weight layer is out of range");
    }
}

} // namespace

void Model::set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor) {
    switch (role) {
    case WeightRole::InEmbed:
        ctx.in_embed = std::move(tensor);
        return;
    case WeightRole::OutEmbed:
        ctx.out_embed = std::move(tensor);
        return;
    case WeightRole::OutNorm:
        ctx.out_norm_w = std::move(tensor);
        return;
    case WeightRole::AttnNorm:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].norm_w = std::move(tensor);
        return;
    case WeightRole::AttnQ_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].q_w = std::move(tensor);
        return;
    case WeightRole::AttnQ_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].q_b = std::move(tensor);
        return;
    case WeightRole::AttnK_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].k_w = std::move(tensor);
        return;
    case WeightRole::AttnK_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].k_b = std::move(tensor);
        return;
    case WeightRole::AttnV_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].v_w = std::move(tensor);
        return;
    case WeightRole::AttnV_B:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].v_b = std::move(tensor);
        return;
    case WeightRole::AttnO_W:
        validate_attn_layer(ctx, layer);
        ctx.attns[layer].o_w = std::move(tensor);
        return;
    case WeightRole::MlpNorm:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].norm_w = std::move(tensor);
        return;
    case WeightRole::MlpGate_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].gate_w = std::move(tensor);
        return;
    case WeightRole::MlpUp_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].up_w = std::move(tensor);
        return;
    case WeightRole::MlpDown_W:
        validate_ffn_layer(ctx, layer);
        ctx.ffns[layer].down_w = std::move(tensor);
        return;
    }
}

} // namespace llaisys::framework
