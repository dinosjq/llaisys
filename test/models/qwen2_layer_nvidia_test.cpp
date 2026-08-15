#include "../../src/models/framework/model_context.hpp"
#include "../../src/models/layers/qwen2/stack.hpp"
#include "../../src/ops/add/op.hpp"
#include "../../src/ops/embedding/op.hpp"
#include "../../src/ops/kv_cache_move/op.hpp"
#include "../../src/ops/linear/op.hpp"
#include "../../src/ops/paged_attention/op.hpp"
#include "../../src/ops/rms_norm/op.hpp"
#include "../../src/ops/rope/op.hpp"
#include "../../src/ops/swiglu/op.hpp"
#include "../../src/tensor/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace {

using llaisys::Tensor;
using llaisys::tensor_t;
using llaisys::framework::ModelContext;

constexpr llaisysDeviceType_t kDevice = LLAISYS_DEVICE_NVIDIA;
constexpr int kDeviceId = 0;
constexpr size_t kTokenNum = 64;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

tensor_t nvidia_f32(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto tensor = Tensor::create(shape, LLAISYS_DTYPE_F32, kDevice, kDeviceId);
    tensor->load(values.data());
    return tensor;
}

tensor_t nvidia_i64(const std::vector<size_t> &shape, const std::vector<int64_t> &values) {
    auto tensor = Tensor::create(shape, LLAISYS_DTYPE_I64, kDevice, kDeviceId);
    tensor->load(values.data());
    return tensor;
}

std::vector<float> values(const tensor_t &tensor) {
    return tensor->view({tensor->numel()})->to(LLAISYS_DEVICE_CPU)->to_vector<float>();
}

void require_close(const tensor_t &actual, const tensor_t &expected, float atol = 2e-4f) {
    const auto actual_values = values(actual);
    const auto expected_values = values(expected);
    require(actual_values.size() == expected_values.size(), "tensor sizes differ");
    for (size_t i = 0; i < actual_values.size(); ++i) {
        require(std::fabs(actual_values[i] - expected_values[i]) <= atol, "tensor values differ");
    }
}

std::vector<float> sequence(size_t count, float scale = 0.01f) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = scale * static_cast<float>((i % 17) + 1);
    }
    return result;
}

ModelContext make_context() {
    ModelContext ctx;
    // The NVIDIA paged-attention implementation accepts head dimensions >= 32.
    ctx.meta = {LLAISYS_DTYPE_F32, 1, 64, 2, 1, 32, 96, 2, 8, 1e-5f, 10000.0f, 0};
    ctx.runtime.token_ids = {1, 2};
    ctx.runtime.pos_ids = {0, 1};
    ctx.runtime.cut_idx = {0, 2};
    ctx.runtime.tot_len = {2};
    ctx.runtime.max_seq_len = 2;

    const auto make = [&](const std::vector<size_t> &shape) {
        return Tensor::create(shape, ctx.meta.dtype, kDevice, kDeviceId);
    };
    ctx.workspace.x = make({2, ctx.meta.hs});
    ctx.workspace.x_norm = make({2, ctx.meta.hs});
    ctx.workspace.q = make({2, ctx.meta.nh * ctx.meta.dh});
    ctx.workspace.k = make({2, ctx.meta.nkvh * ctx.meta.dh});
    ctx.workspace.v = make({2, ctx.meta.nkvh * ctx.meta.dh});
    ctx.workspace.q_rope = make({2, ctx.meta.nh, ctx.meta.dh});
    ctx.workspace.k_rope = make({2, ctx.meta.nkvh, ctx.meta.dh});
    ctx.workspace.attn_val = make({2, ctx.meta.nh, ctx.meta.dh});
    ctx.workspace.attn_out = make({2, ctx.meta.hs});
    ctx.workspace.x_attn = make({2, ctx.meta.hs});
    ctx.workspace.m_norm = make({2, ctx.meta.hs});
    ctx.workspace.gate = make({2, ctx.meta.di});
    ctx.workspace.up = make({2, ctx.meta.di});
    ctx.workspace.swiglu = make({2, ctx.meta.di});
    ctx.workspace.down = make({2, ctx.meta.hs});
    ctx.workspace.x_mlp = make({2, ctx.meta.hs});
    ctx.workspace.logits = make({1, ctx.meta.voc});
    ctx.workspace.x_part = make({1, ctx.meta.hs});
    ctx.workspace.x_part_norm = make({1, ctx.meta.hs});
    ctx.workspace.block_ids = nvidia_i64({1, 2}, {1, 0});
    ctx.workspace.cut_idx = nvidia_i64({1 + 1}, ctx.runtime.cut_idx);
    ctx.workspace.tot_len = nvidia_i64({1}, ctx.runtime.tot_len);
    ctx.workspace.pos_ids = nvidia_i64({2}, ctx.runtime.pos_ids);

    ctx.k_caches = {make({1, kTokenNum, ctx.meta.nkvh, ctx.meta.dh})};
    ctx.v_caches = {make({1, kTokenNum, ctx.meta.nkvh, ctx.meta.dh})};
    ctx.attns.resize(1);
    ctx.ffns.resize(1);
    ctx.in_embed = nvidia_f32({8, ctx.meta.hs}, sequence(8 * ctx.meta.hs));
    ctx.out_norm_w = nvidia_f32({ctx.meta.hs}, sequence(ctx.meta.hs));
    ctx.out_embed = nvidia_f32({ctx.meta.voc, ctx.meta.hs}, sequence(ctx.meta.voc * ctx.meta.hs));

    auto &attn = ctx.attns[0];
    attn.norm_w = nvidia_f32({ctx.meta.hs}, sequence(ctx.meta.hs, 0.02f));
    attn.q_w = nvidia_f32({ctx.meta.nh * ctx.meta.dh, ctx.meta.hs}, sequence(ctx.meta.nh * ctx.meta.dh * ctx.meta.hs));
    attn.q_b = nvidia_f32({ctx.meta.nh * ctx.meta.dh}, sequence(ctx.meta.nh * ctx.meta.dh, 0.001f));
    attn.k_w = nvidia_f32({ctx.meta.nkvh * ctx.meta.dh, ctx.meta.hs}, sequence(ctx.meta.nkvh * ctx.meta.dh * ctx.meta.hs));
    attn.k_b = nvidia_f32({ctx.meta.nkvh * ctx.meta.dh}, sequence(ctx.meta.nkvh * ctx.meta.dh, 0.001f));
    attn.v_w = nvidia_f32({ctx.meta.nkvh * ctx.meta.dh, ctx.meta.hs}, sequence(ctx.meta.nkvh * ctx.meta.dh * ctx.meta.hs));
    attn.v_b = nvidia_f32({ctx.meta.nkvh * ctx.meta.dh}, sequence(ctx.meta.nkvh * ctx.meta.dh, 0.001f));
    attn.o_w = nvidia_f32({ctx.meta.hs, ctx.meta.nh * ctx.meta.dh}, sequence(ctx.meta.hs * ctx.meta.nh * ctx.meta.dh));

    auto &ffn = ctx.ffns[0];
    ffn.norm_w = nvidia_f32({ctx.meta.hs}, sequence(ctx.meta.hs, 0.02f));
    ffn.gate_w = nvidia_f32({ctx.meta.di, ctx.meta.hs}, sequence(ctx.meta.di * ctx.meta.hs));
    ffn.up_w = nvidia_f32({ctx.meta.di, ctx.meta.hs}, sequence(ctx.meta.di * ctx.meta.hs, 0.015f));
    ffn.down_w = nvidia_f32({ctx.meta.hs, ctx.meta.di}, sequence(ctx.meta.hs * ctx.meta.di));
    return ctx;
}

// Copy of src/models/qwen2.cpp:405-466 (current production attention and output order).
void legacy_qwen2_forward_copy(ModelContext &ctx, bool is_prefill) {
    auto token_ids = nvidia_i64({ctx.runtime.token_ids.size()}, ctx.runtime.token_ids);
    llaisys::ops::embedding(ctx.workspace.x, token_ids, ctx.in_embed);
    const float scale = 1.0f / std::sqrt(static_cast<float>(ctx.meta.dh));

    for (size_t layer = 0; layer < ctx.meta.nlayer; ++layer) {
        const auto &attn = ctx.attns[layer];
        llaisys::ops::rms_norm(ctx.workspace.x_norm, ctx.workspace.x, attn.norm_w, ctx.meta.epsilon);
        llaisys::ops::linear(ctx.workspace.q, ctx.workspace.x_norm, attn.q_w, attn.q_b);
        llaisys::ops::linear(ctx.workspace.k, ctx.workspace.x_norm, attn.k_w, attn.k_b);
        llaisys::ops::linear(ctx.workspace.v, ctx.workspace.x_norm, attn.v_w, attn.v_b);
        llaisys::ops::rope(ctx.workspace.q_rope, ctx.workspace.q->view({2, ctx.meta.nh, ctx.meta.dh}), ctx.workspace.pos_ids, ctx.meta.theta);
        llaisys::ops::rope(ctx.workspace.k_rope, ctx.workspace.k->view({2, ctx.meta.nkvh, ctx.meta.dh}), ctx.workspace.pos_ids, ctx.meta.theta);
        llaisys::ops::kv_cache_move(ctx.k_caches[layer], ctx.workspace.k_rope, ctx.workspace.block_ids, ctx.workspace.cut_idx, ctx.workspace.pos_ids,
                                    ctx.runtime.max_seq_len);
        llaisys::ops::kv_cache_move(ctx.v_caches[layer], ctx.workspace.v->view({2, ctx.meta.nkvh, ctx.meta.dh}), ctx.workspace.block_ids, ctx.workspace.cut_idx,
                                    ctx.workspace.pos_ids, ctx.runtime.max_seq_len);
        llaisys::ops::paged_attention(ctx.workspace.attn_val, ctx.workspace.q_rope, ctx.k_caches[layer], ctx.v_caches[layer], ctx.workspace.block_ids,
                                      ctx.workspace.cut_idx, ctx.workspace.tot_len, ctx.runtime.max_seq_len, 1, scale, is_prefill);
        llaisys::ops::linear(ctx.workspace.attn_out, ctx.workspace.attn_val->view({2, ctx.meta.hs}), attn.o_w, nullptr);
        llaisys::ops::add(ctx.workspace.x_attn, ctx.workspace.x, ctx.workspace.attn_out);
        std::swap(ctx.workspace.x, ctx.workspace.x_attn);
        ctx.workspace.capture_post_attn0 = ctx.workspace.x->to(kDevice, kDeviceId);

        const auto &ffn = ctx.ffns[layer];
        llaisys::ops::rms_norm(ctx.workspace.m_norm, ctx.workspace.x, ffn.norm_w, ctx.meta.epsilon);
        llaisys::ops::linear(ctx.workspace.gate, ctx.workspace.m_norm, ffn.gate_w, nullptr);
        llaisys::ops::linear(ctx.workspace.up, ctx.workspace.m_norm, ffn.up_w, nullptr);
        llaisys::ops::swiglu(ctx.workspace.swiglu, ctx.workspace.gate, ctx.workspace.up);
        llaisys::ops::linear(ctx.workspace.down, ctx.workspace.swiglu, ffn.down_w, nullptr);
        llaisys::ops::add(ctx.workspace.x_mlp, ctx.workspace.x, ctx.workspace.down);
        std::swap(ctx.workspace.x, ctx.workspace.x_mlp);
        ctx.workspace.capture_post_ffn0 = ctx.workspace.x->to(kDevice, kDeviceId);
    }

    auto last_token = nvidia_i64({1}, {1});
    llaisys::ops::embedding(ctx.workspace.x_part, last_token, ctx.workspace.x);
    llaisys::ops::rms_norm(ctx.workspace.x_part_norm, ctx.workspace.x_part, ctx.out_norm_w, ctx.meta.epsilon);
    llaisys::ops::linear(ctx.workspace.logits, ctx.workspace.x_part_norm, ctx.out_embed, nullptr);
}

void test_stack_matches_production_capture_points() {
    auto actual = make_context();
    auto expected = make_context();
    llaisys::framework::run_qwen2_layer_stack(actual, true);
    legacy_qwen2_forward_copy(expected, true);

    require_close(actual.workspace.capture_post_attn0, expected.workspace.capture_post_attn0);
    require_close(actual.workspace.capture_post_ffn0, expected.workspace.capture_post_ffn0);
    require_close(actual.workspace.logits, expected.workspace.logits);
}

} // namespace

int main() {
    test_stack_matches_production_capture_points();
    std::cout << "qwen2 NVIDIA layer stack capture test passed\n";
    return 0;
}
