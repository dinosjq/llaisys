#include "ffn.hpp"

#include "../../../ops/add/op.hpp"
#include "../../../ops/linear/op.hpp"
#include "../../../ops/rms_norm/op.hpp"
#include "../../../ops/swiglu/op.hpp"

#include <stdexcept>
#include <utility>

namespace llaisys::model {

Qwen2FFN::Qwen2FFN(const FfnWeights *weights) : _w(weights) {}

void Qwen2FFN::forward(ModelContext &ctx) {
    forward(static_cast<Qwen2Context &>(ctx));
}

void Qwen2FFN::forward(Qwen2Context &ctx) {
    if (this->_w == nullptr) {
        throw std::invalid_argument("Qwen2FFN requires weights");
    }

    auto &ws = ctx.workspace;
    const auto &w = *this->_w;

    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, ctx.meta.epsilon);
    ops::linear(ws.gate, ws.m_norm, w.gate_w, nullptr);
    ops::linear(ws.up, ws.m_norm, w.up_w, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    ops::linear(ws.down, ws.swiglu, w.down_w, nullptr);

    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

} // namespace llaisys::model
