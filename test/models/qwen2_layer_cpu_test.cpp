#include "../../src/models/core/model_context.hpp"
#include "../../src/layers/qwen2/qwen2.hpp"
#include "../../src/layers/qwen2/qwen2.hpp"
#include "../../src/ops/add/op.hpp"
#include "../../src/ops/embedding/op.hpp"
#include "../../src/ops/linear/op.hpp"
#include "../../src/ops/rms_norm/op.hpp"
#include "../../src/ops/swiglu/op.hpp"
#include "../../src/tensor/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

using llaisys::Tensor;
using llaisys::tensor_t;
using llaisys::framework::ModelContext;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

tensor_t cpu_f32(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto tensor = Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    tensor->load(values.data());
    return tensor;
}

void require_close(const tensor_t &actual, const tensor_t &expected, float atol = 1e-5f) {
    require(actual->numel() == expected->numel(), "tensor sizes differ");
    const auto *actual_values = reinterpret_cast<const float *>(actual->data());
    const auto *expected_values = reinterpret_cast<const float *>(expected->data());
    for (size_t i = 0; i < actual->numel(); ++i) {
        require(std::fabs(actual_values[i] - expected_values[i]) <= atol, "tensor values differ");
    }
}

void legacy_ffn_block(ModelContext &ctx, const llaisys::framework::FfnWeights &weights) {
    llaisys::ops::rms_norm(ctx.workspace.m_norm, ctx.workspace.x, weights.norm_w, ctx.meta.epsilon);
    llaisys::ops::linear(ctx.workspace.gate, ctx.workspace.m_norm, weights.gate_w, nullptr);
    llaisys::ops::linear(ctx.workspace.up, ctx.workspace.m_norm, weights.up_w, nullptr);
    llaisys::ops::swiglu(ctx.workspace.swiglu, ctx.workspace.gate, ctx.workspace.up);
    llaisys::ops::linear(ctx.workspace.down, ctx.workspace.swiglu, weights.down_w, nullptr);
    llaisys::ops::add(ctx.workspace.x_mlp, ctx.workspace.x, ctx.workspace.down);
    std::swap(ctx.workspace.x, ctx.workspace.x_mlp);
}

ModelContext make_ffn_context() {
    ModelContext ctx;
    ctx.meta.dtype = LLAISYS_DTYPE_F32;
    ctx.meta.hs = 4;
    ctx.meta.di = 8;
    ctx.meta.epsilon = 1e-5f;
    ctx.ffns.resize(1);

    const std::vector<float> x_values{
        0.1f, -0.2f, 0.3f, -0.4f,
        0.5f, -0.6f, 0.7f, -0.8f,
    };
    ctx.workspace.x = cpu_f32({2, ctx.meta.hs}, x_values);
    ctx.workspace.x_mlp = Tensor::create({2, ctx.meta.hs}, LLAISYS_DTYPE_F32);
    ctx.workspace.m_norm = Tensor::create({2, ctx.meta.hs}, LLAISYS_DTYPE_F32);
    ctx.workspace.gate = Tensor::create({2, ctx.meta.di}, LLAISYS_DTYPE_F32);
    ctx.workspace.up = Tensor::create({2, ctx.meta.di}, LLAISYS_DTYPE_F32);
    ctx.workspace.swiglu = Tensor::create({2, ctx.meta.di}, LLAISYS_DTYPE_F32);
    ctx.workspace.down = Tensor::create({2, ctx.meta.hs}, LLAISYS_DTYPE_F32);

    ctx.ffns[0].norm_w = cpu_f32({ctx.meta.hs}, {1.0f, 0.9f, 1.1f, 0.8f});
    ctx.ffns[0].gate_w = cpu_f32(
        {ctx.meta.di, ctx.meta.hs},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f,
         1.7f, 1.8f, 1.9f, 2.0f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f, 2.7f, 2.8f, 2.9f, 3.0f, 3.1f, 3.2f});
    ctx.ffns[0].up_w = cpu_f32(
        {ctx.meta.di, ctx.meta.hs},
        {0.2f, 0.1f, 0.4f, 0.3f, 0.6f, 0.5f, 0.8f, 0.7f, 1.0f, 0.9f, 1.2f, 1.1f, 1.4f, 1.3f, 1.6f, 1.5f,
         1.8f, 1.7f, 2.0f, 1.9f, 2.2f, 2.1f, 2.4f, 2.3f, 2.6f, 2.5f, 2.8f, 2.7f, 3.0f, 2.9f, 3.2f, 3.1f});
    ctx.ffns[0].down_w = cpu_f32(
        {ctx.meta.hs, ctx.meta.di},
        {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f, 0.2f, 0.1f,
         0.3f, 0.5f, 0.7f, 0.9f, 0.9f, 0.7f, 0.5f, 0.3f, 0.2f, 0.4f, 0.6f, 0.8f, 0.8f, 0.6f, 0.4f, 0.2f});
    return ctx;
}

void test_embedding_matches_ops_embedding() {
    ModelContext ctx;
    ctx.workspace.x = Tensor::create({3, 2}, LLAISYS_DTYPE_F32);
    ctx.in_embed = cpu_f32({4, 2}, {0.0f, 0.1f, 1.0f, 1.1f, 2.0f, 2.1f, 3.0f, 3.1f});
    ctx.runtime.token_ids = {3, 1, 2};

    auto token_ids = Tensor::create({3}, LLAISYS_DTYPE_I64);
    token_ids->load(ctx.runtime.token_ids.data());
    auto expected = Tensor::create({3, 2}, LLAISYS_DTYPE_F32);
    llaisys::ops::embedding(expected, token_ids, ctx.in_embed);

    llaisys::framework::Qwen2Embedding{}.forward(ctx);
    require_close(ctx.workspace.x, expected);
}

void test_ffn_matches_legacy_block() {
    auto actual = make_ffn_context();
    auto expected = make_ffn_context();

    llaisys::framework::Qwen2FFN(&actual.ffns[0]).forward(actual);
    legacy_ffn_block(expected, expected.ffns[0]);

    require_close(actual.workspace.x, expected.workspace.x);
}

} // namespace

int main() {
    test_embedding_matches_ops_embedding();
    test_ffn_matches_legacy_block();
    std::cout << "qwen2 embedding and ffn CPU layer tests passed\n";
    return 0;
}
