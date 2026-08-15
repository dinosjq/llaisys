#include "../../src/config.hpp"
#include "../../src/llaisys/llaisys_tensor.hpp"
#include "../../src/models/qwen2.hpp"
#include "../../src/tensor/tensor.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace {

using llaisys::Qwen2;
using llaisys::Qwen2Meta;
using llaisys::Qwen2Pack;
using llaisys::Tensor;
using llaisys::tensor_t;

constexpr llaisysDeviceType_t kDevice = LLAISYS_DEVICE_NVIDIA;
constexpr int kDeviceId = 0;
constexpr float kFixtureScale = 0.01f;

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

std::vector<float> sequence(size_t count, float scale = 0.01f) {
    std::vector<float> result(count);
    for (size_t i = 0; i < count; ++i) {
        result[i] = kFixtureScale * scale * static_cast<float>((i % 17) + 1);
    }
    return result;
}

tensor_t nvidia_f32(const std::vector<size_t> &shape, const std::vector<float> &values) {
    auto tensor = Tensor::create(shape, LLAISYS_DTYPE_F32, kDevice, kDeviceId);
    tensor->load(values.data());
    return tensor;
}

llaisysTensor_t wrap(tensor_t tensor) {
    return new LlaisysTensor{std::move(tensor)};
}

std::vector<float> values(const tensor_t &tensor) {
    return tensor->view({tensor->numel()})->to(LLAISYS_DEVICE_CPU)->to_vector<float>();
}

void require_close(const tensor_t &actual, const tensor_t &expected, const char *message, float atol = 2e-4f,
                   float rtol = 2e-2f) {
    const auto actual_values = values(actual);
    const auto expected_values = values(expected);
    require(actual_values.size() == expected_values.size(), message);
    for (size_t i = 0; i < actual_values.size(); ++i) {
        const float tolerance = atol + rtol * std::fabs(expected_values[i]);
        if (!std::isfinite(actual_values[i]) || !std::isfinite(expected_values[i])
            || std::fabs(actual_values[i] - expected_values[i]) > tolerance) {
            std::ostringstream detail;
            detail << message << " at index " << i << ": " << actual_values[i] << " != " << expected_values[i];
            throw std::runtime_error(detail.str());
        }
    }
}

Qwen2Meta tiny_meta() {
    // Matches qwen2_layer_nvidia_test fixture: nh*dh=hs, dh>=32 for NVIDIA paged attention.
    return Qwen2Meta{LLAISYS_DTYPE_F32, 2, 64, 2, 1, 32, 96, 4, 8, 1e-5f, 10000.0f, 0};
}

void fill_tiny_weights(Qwen2 &model) {
    auto &w = model.weights();
    const auto &meta = tiny_meta();
    w.in_embed = wrap(nvidia_f32({meta.voc, meta.hs}, sequence(meta.voc * meta.hs)));
    w.out_norm_w = wrap(nvidia_f32({meta.hs}, sequence(meta.hs)));
    w.out_embed = wrap(nvidia_f32({meta.voc, meta.hs}, sequence(meta.voc * meta.hs)));
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        w.attn_norm_w[layer] = wrap(nvidia_f32({meta.hs}, sequence(meta.hs, 0.02f)));
        w.attn_q_w[layer] = wrap(nvidia_f32({meta.nh * meta.dh, meta.hs}, sequence(meta.nh * meta.dh * meta.hs)));
        w.attn_q_b[layer] = wrap(nvidia_f32({meta.nh * meta.dh}, sequence(meta.nh * meta.dh, 0.001f)));
        w.attn_k_w[layer] = wrap(nvidia_f32({meta.nkvh * meta.dh, meta.hs}, sequence(meta.nkvh * meta.dh * meta.hs)));
        w.attn_k_b[layer] = wrap(nvidia_f32({meta.nkvh * meta.dh}, sequence(meta.nkvh * meta.dh, 0.001f)));
        w.attn_v_w[layer] = wrap(nvidia_f32({meta.nkvh * meta.dh, meta.hs}, sequence(meta.nkvh * meta.dh * meta.hs)));
        w.attn_v_b[layer] = wrap(nvidia_f32({meta.nkvh * meta.dh}, sequence(meta.nkvh * meta.dh, 0.001f)));
        w.attn_o_w[layer] = wrap(nvidia_f32({meta.hs, meta.nh * meta.dh}, sequence(meta.hs * meta.nh * meta.dh)));
        w.mlp_norm_w[layer] = wrap(nvidia_f32({meta.hs}, sequence(meta.hs, 0.02f)));
        w.mlp_gate_w[layer] = wrap(nvidia_f32({meta.di, meta.hs}, sequence(meta.di * meta.hs)));
        w.mlp_up_w[layer] = wrap(nvidia_f32({meta.di, meta.hs}, sequence(meta.di * meta.hs, 0.015f)));
        w.mlp_down_w[layer] = wrap(nvidia_f32({meta.hs, meta.di}, sequence(meta.hs * meta.di)));
    }
}

std::unique_ptr<Qwen2> make_model() {
    int device_id = kDeviceId;
    auto model = std::make_unique<Qwen2>(tiny_meta(), kDevice, &device_id, 1);
    model->stop();
    fill_tiny_weights(*model);
    // Dual-path test always exercises layer path explicitly; default remains legacy.
    model->set_use_layer_forward(true);
    require(model->use_layer_forward(), "layer forward flag should be true when enabled");
    return model;
}

Qwen2Pack make_prefill_pack(std::mt19937 &rng) {
    Qwen2Pack pack;
    pack.token_ids = {1, 2};
    pack.pos_ids = {0, 1};
    pack.cut_idx = {0, 2};
    pack.tot_len = {2};
    pack.top_k = {1};
    pack.top_p = {1.0f};
    pack.temperature = {0.0f};
    pack.rngs = {&rng};
    pack.max_seq_len = 2;
    return pack;
}

Qwen2Pack make_decode_pack(std::mt19937 &rng) {
    Qwen2Pack pack;
    pack.token_ids = {3};
    pack.pos_ids = {2};
    pack.cut_idx = {0, 1};
    pack.tot_len = {3};
    pack.top_k = {1};
    pack.top_p = {1.0f};
    pack.temperature = {0.0f};
    pack.rngs = {&rng};
    pack.max_seq_len = 1;
    return pack;
}

std::vector<int64_t> make_block_ids() {
    // prepare_block_table layout: [cumulative_block_count, block_id_0, ...]
    std::vector<int64_t> block_ids(MAX_BLOCK_NUM, -1);
    block_ids[0] = 1;
    block_ids[1] = 0;
    return block_ids;
}

void test_prefill_layer_matches_legacy() {
    auto legacy = make_model();
    auto layered = make_model();
    std::mt19937 rng_a(7);
    std::mt19937 rng_b(7);
    auto pack_a = make_prefill_pack(rng_a);
    auto pack_b = make_prefill_pack(rng_b);
    auto block_ids_a = make_block_ids();
    auto block_ids_b = make_block_ids();

    const auto tokens_legacy = legacy->forward_legacy(pack_a, block_ids_a, true);
    const auto tokens_layers = layered->forward_layers(pack_b, block_ids_b, true);

    require(tokens_legacy.size() == 1 && tokens_layers.size() == 1, "prefill should emit one greedy token");
    require(tokens_legacy[0] == tokens_layers[0], "prefill greedy tokens differ between legacy and layer path");
    require_close(layered->last_logits(1), legacy->last_logits(1), "prefill logits differ");
}

void test_decode_after_prefill_layer_matches_legacy() {
    auto legacy = make_model();
    auto layered = make_model();
    std::mt19937 rng_a(11);
    std::mt19937 rng_b(11);
    auto prefill_a = make_prefill_pack(rng_a);
    auto prefill_b = make_prefill_pack(rng_b);
    auto block_ids_a = make_block_ids();
    auto block_ids_b = make_block_ids();

    (void)legacy->forward_legacy(prefill_a, block_ids_a, true);
    (void)layered->forward_layers(prefill_b, block_ids_b, true);

    auto decode_a = make_decode_pack(rng_a);
    auto decode_b = make_decode_pack(rng_b);
    const auto tokens_legacy = legacy->forward_legacy(decode_a, block_ids_a, false);
    const auto tokens_layers = layered->forward_layers(decode_b, block_ids_b, false);

    require(tokens_legacy.size() == 1 && tokens_layers.size() == 1, "decode should emit one greedy token");
    require(tokens_legacy[0] == tokens_layers[0], "decode greedy tokens differ between legacy and layer path");
    require_close(layered->last_logits(1), legacy->last_logits(1), "decode logits differ");
}

void test_default_flag_is_legacy() {
    unsetenv("LLAISYS_QWEN2_LAYER_FORWARD");
    int device_id = kDeviceId;
    Qwen2 model(tiny_meta(), kDevice, &device_id, 1);
    model.stop();
    require(!model.use_layer_forward(), "default use_layer_forward_ must be false");
}

} // namespace

int main() {
    test_default_flag_is_legacy();
    test_prefill_layer_matches_legacy();
    test_decode_after_prefill_layer_matches_legacy();
    std::cout << "qwen2 dual path NVIDIA test passed\n";
    return 0;
}
