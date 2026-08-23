#include "ffn.hpp"

#include "../../weight/qwen2_w8_weights.hpp"
#include "../../../../model/layer/utils/linear_dispatch.hpp"
#include "../../../../ops/add/op.hpp"
#include "../../../../ops/rms_norm/op.hpp"
#include "../../../../ops/swiglu/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

Qwen2FFN::Qwen2FFN(QwenFfnWeights &weights) : _w(weights) {}

void Qwen2FFN::forward(ModelContext &ctx) {
    forward(static_cast<Qwen2Context &>(ctx));
}

void Qwen2FFN::forward(Qwen2Context &ctx) {
    auto &ws = ctx.current();
    const auto &w = this->_w;
    const auto *w8 = dynamic_cast<const QwenW8FfnWeights *>(&w);
    const auto &meta = ctx.meta;

    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, meta.epsilon);
    layers::utils::linear_proj(ws.gate, ws.m_norm, w.gate_w, w8 ? w8->gate_scale : nullptr, nullptr);
    layers::utils::linear_proj(ws.up, ws.m_norm, w.up_w, w8 ? w8->up_scale : nullptr, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    layers::utils::linear_proj(ws.down, ws.swiglu, w.down_w, w8 ? w8->down_scale : nullptr, nullptr);

    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

} // namespace llaisys::model
