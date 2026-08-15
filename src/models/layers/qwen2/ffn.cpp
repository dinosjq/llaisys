#include "ffn.hpp"

#include "../../../ops/add/op.hpp"
#include "../../../ops/linear/op.hpp"
#include "../../../ops/rms_norm/op.hpp"
#include "../../../ops/swiglu/op.hpp"

#include <utility>

namespace llaisys::framework {

Qwen2FFN::Qwen2FFN(const FfnWeights *weights, size_t layer_index_for_debug) : w_(weights), layer_index_for_debug_(layer_index_for_debug) {}

void Qwen2FFN::forward(ModelContext &ctx) {
    auto &ws = ctx.workspace;
    const auto &w = *w_;

    // 多层感知机 MLP block
    ops::rms_norm(ws.m_norm, ws.x, w.norm_w, ctx.meta.epsilon);
    ops::linear(ws.gate, ws.m_norm, w.gate_w, nullptr);
    ops::linear(ws.up, ws.m_norm, w.up_w, nullptr);
    ops::swiglu(ws.swiglu, ws.gate, ws.up);
    ops::linear(ws.down, ws.swiglu, w.down_w, nullptr);

    // 残差连接: x + MLP(x)
    ops::add(ws.x_mlp, ws.x, ws.down);
    std::swap(ws.x, ws.x_mlp);
}

} // namespace llaisys::framework
