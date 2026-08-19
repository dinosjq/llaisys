#include "ffn.hpp"

#include "../../../qwen2/meta/qwen2_meta.hpp"
#include "../../../qwen2/weight/qwen2_w8_weight.hpp"
#include "../../../../model/layer/utils/linear_dispatch.hpp"
#include "../../../../ops/add/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/swiglu/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

LlamaFFN::LlamaFFN(llama_ffn_weights_t weights) : _w(std::move(weights)) {}

void LlamaFFN::forward(ModelContext &ctx) {
    forward(static_cast<LlamaContext &>(ctx));
}

void LlamaFFN::forward(LlamaContext &ctx) {
    if (this->_w == nullptr) {
        throw std::invalid_argument("LlamaFFN requires weights");
    }

    auto &ws = ctx.workspace;
    const auto &w = *this->_w;
    const auto *w8 = dynamic_cast<const QwenW8FfnWeights *>(this->_w.get());

    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, qwen_meta(ctx).epsilon);
    layers::utils::linear_proj(ws.gate, ws.m_norm, w.gate_w, w8 ? w8->gate_scale : nullptr, nullptr);
    layers::utils::linear_proj(ws.up, ws.m_norm, w.up_w, w8 ? w8->up_scale : nullptr, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    layers::utils::linear_proj(ws.down, ws.swiglu, w.down_w, w8 ? w8->down_scale : nullptr, nullptr);

    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

} // namespace llaisys::model
