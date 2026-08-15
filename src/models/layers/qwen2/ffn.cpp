#include "ffn.hpp"

#include "../../../ops/add/op.hpp"
#include "../../../ops/linear/op.hpp"
#include "../../../ops/rms_norm/op.hpp"
#include "../../../ops/swiglu/op.hpp"

#include <utility>

namespace llaisys::framework {

Qwen2FFN::Qwen2FFN(const FfnWeights *weights, size_t layer_index_for_debug) : w_(weights), layer_index_for_debug_(layer_index_for_debug) {}

void Qwen2FFN::forward(ModelContext &ctx) {
    ops::rms_norm(ctx.workspace.m_norm, ctx.workspace.x, w_->norm_w, ctx.meta.epsilon);
    ops::linear(ctx.workspace.gate, ctx.workspace.m_norm, w_->gate_w, nullptr);
    ops::linear(ctx.workspace.up, ctx.workspace.m_norm, w_->up_w, nullptr);
    ops::swiglu(ctx.workspace.swiglu, ctx.workspace.gate, ctx.workspace.up);
    ops::linear(ctx.workspace.down, ctx.workspace.swiglu, w_->down_w, nullptr);
    ops::add(ctx.workspace.x_mlp, ctx.workspace.x, ctx.workspace.down);
    std::swap(ctx.workspace.x, ctx.workspace.x_mlp);
}

} // namespace llaisys::framework
