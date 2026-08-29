#include "ffn.hpp"

#include "../../../qwen2/weight/qwen2_w8_weights.hpp"
#include "../../../../model/layer/utils/linear_dispatch.hpp"
#include "../../../../ops/add/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/rms_norm_quant/op.hpp"
#include "../../../../ops/swiglu/op.hpp"
#include "../../../../ops/swiglu_quant/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

LlamaFFN::LlamaFFN(QwenFfnWeights &weights) : _w(weights) {}

void LlamaFFN::forward(ModelContext &ctx) {
    forward(static_cast<LlamaContext &>(ctx));
}

void LlamaFFN::forward(LlamaContext &ctx) {
    auto &ws = ctx.current();
    const auto &w = this->_w;
    const auto *w8 = dynamic_cast<const QwenW8FfnWeights *>(&w);
    const auto &meta = ctx.meta;

    if (w8) {
        ops::rms_norm_quant(ws.m_norm_q, ws.m_norm_a_scale, ws.x, w.norm_w, meta.epsilon);
    } else {
        ops::rms_norm(ws.m_norm, ws.x, w.norm_w, meta.epsilon);
    }
    const tensor_t ff_in = w8 ? ws.m_norm_q : ws.m_norm;
    const tensor_t ff_scale = w8 ? ws.m_norm_a_scale : tensor_t();
    layers::utils::linear_proj(ws.gate, ff_in, ff_scale, w.gate_w, w8 ? w8->gate_scale : nullptr, nullptr);
    layers::utils::linear_proj(ws.up, ff_in, ff_scale, w.up_w, w8 ? w8->up_scale : nullptr, nullptr);
    if (w8) {
        ops::swiglu_quant(ws.swiglu_q, ws.swiglu_a_scale, ws.gate, ws.up);
    } else {
        ops::swiglu(ws.swiglu, ws.gate, ws.up);
    }
    layers::utils::linear_proj(ws.down, w8 ? ws.swiglu_q : ws.swiglu, w8 ? ws.swiglu_a_scale : tensor_t(),
                               w.down_w, w8 ? w8->down_scale : nullptr, nullptr);

    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

} // namespace llaisys::model
