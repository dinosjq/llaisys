#include "../../src/config.hpp"
#include "../../src/models/qwen2/qwen2.hpp"
#include "../../src/tensor/tensor.hpp"
#include "../../src/utils/byte_map.hpp"

#include <cmath>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

using llaisys::Qwen2;
using llaisys::model::BatchPack;
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
    auto host = Tensor::create(shape, LLAISYS_DTYPE_F32, LLAISYS_DEVICE_CPU, 0);
    host->load(values.data());
    return host->to(kDevice, kDeviceId);
}

void fill_tiny_weights(Qwen2 &model) {
    const auto &meta = model.meta();
    auto layer_name = [](size_t layer, const char *suffix) {
        return "model.layers." + std::to_string(layer) + "." + suffix;
    };
    model.set_weight("model.embed_tokens.weight", nvidia_f32({meta.voc, meta.hs}, sequence(meta.voc * meta.hs)));
    model.set_weight("lm_head.weight", nvidia_f32({meta.voc, meta.hs}, sequence(meta.voc * meta.hs, 0.02f)));
    model.set_weight("model.norm.weight", nvidia_f32({meta.hs}, sequence(meta.hs)));
    for (size_t layer = 0; layer < meta.nlayer; ++layer) {
        model.set_weight(layer_name(layer, "input_layernorm.weight"), nvidia_f32({meta.hs}, sequence(meta.hs)));
        model.set_weight(layer_name(layer, "self_attn.q_proj.weight"),
                         nvidia_f32({meta.nh * meta.dh, meta.hs}, sequence(meta.nh * meta.dh * meta.hs)));
        model.set_weight(layer_name(layer, "self_attn.q_proj.bias"),
                         nvidia_f32({meta.nh * meta.dh}, sequence(meta.nh * meta.dh, 0.001f)));
        model.set_weight(layer_name(layer, "self_attn.k_proj.weight"),
                         nvidia_f32({meta.nkvh * meta.dh, meta.hs}, sequence(meta.nkvh * meta.dh * meta.hs)));
        model.set_weight(layer_name(layer, "self_attn.k_proj.bias"),
                         nvidia_f32({meta.nkvh * meta.dh}, sequence(meta.nkvh * meta.dh, 0.001f)));
        model.set_weight(layer_name(layer, "self_attn.v_proj.weight"),
                         nvidia_f32({meta.nkvh * meta.dh, meta.hs}, sequence(meta.nkvh * meta.dh * meta.hs)));
        model.set_weight(layer_name(layer, "self_attn.v_proj.bias"),
                         nvidia_f32({meta.nkvh * meta.dh}, sequence(meta.nkvh * meta.dh, 0.001f)));
        model.set_weight(layer_name(layer, "self_attn.o_proj.weight"),
                         nvidia_f32({meta.hs, meta.nh * meta.dh}, sequence(meta.hs * meta.nh * meta.dh)));
        model.set_weight(layer_name(layer, "post_attention_layernorm.weight"),
                         nvidia_f32({meta.hs}, sequence(meta.hs, 0.02f)));
        model.set_weight(layer_name(layer, "mlp.gate_proj.weight"),
                         nvidia_f32({meta.di, meta.hs}, sequence(meta.di * meta.hs)));
        model.set_weight(layer_name(layer, "mlp.up_proj.weight"),
                         nvidia_f32({meta.di, meta.hs}, sequence(meta.di * meta.hs, 0.015f)));
        model.set_weight(layer_name(layer, "mlp.down_proj.weight"),
                         nvidia_f32({meta.hs, meta.di}, sequence(meta.hs * meta.di)));
    }
}

std::unique_ptr<Qwen2> make_model() {
    int device_id = kDeviceId;
    auto model = std::make_unique<Qwen2>(kDevice, &device_id, 1);
    llaisys::utils::ByteMap kv;
    llaisys::utils::put_string(kv, "torch_dtype", "float32");
    llaisys::utils::put_as<int64_t>(kv, "num_hidden_layers", 1);
    llaisys::utils::put_as<int64_t>(kv, "hidden_size", 64);
    llaisys::utils::put_as<int64_t>(kv, "num_attention_heads", 2);
    llaisys::utils::put_as<int64_t>(kv, "num_key_value_heads", 2);
    llaisys::utils::put_as<int64_t>(kv, "head_dim", 32);
    llaisys::utils::put_as<int64_t>(kv, "intermediate_size", 128);
    llaisys::utils::put_as<int64_t>(kv, "max_position_embeddings", 16);
    llaisys::utils::put_as<int64_t>(kv, "vocab_size", 32);
    llaisys::utils::put_as<float>(kv, "rms_norm_eps", 1e-5f);
    llaisys::utils::put_as<float>(kv, "rope_theta", 10000.f);
    llaisys::utils::put_as<int64_t>(kv, "eos_token_id", 0);
    model->set_meta(kv);
    model->stop();
    fill_tiny_weights(*model);
    return model;
}

BatchPack make_prefill_pack(std::mt19937 &rng) {
    BatchPack pack;
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

BatchPack make_decode_pack(std::mt19937 &rng) {
    BatchPack pack;
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
    std::vector<int64_t> block_ids(MAX_BLOCK_NUM, -1);
    block_ids[0] = 1;
    block_ids[1] = 0;
    return block_ids;
}

void test_layer_prefill_and_decode() {
    auto model = make_model();
    std::mt19937 rng(7);
    auto pack = make_prefill_pack(rng);
    auto block_ids = make_block_ids();
    const auto tokens_prefill = model->forward(pack, block_ids, true);
    require(tokens_prefill.size() == 1, "prefill should emit one greedy token");
    require(tokens_prefill[0] >= 0 && tokens_prefill[0] < static_cast<int64_t>(model->meta().voc),
            "prefill token should be in vocab");

    auto decode = make_decode_pack(rng);
    const auto tokens_decode = model->forward(decode, block_ids, false);
    require(tokens_decode.size() == 1, "decode should emit one greedy token");
    require(tokens_decode[0] >= 0 && tokens_decode[0] < static_cast<int64_t>(model->meta().voc),
            "decode token should be in vocab");
    auto logits = model->last_logits(1);
    require(logits != nullptr && logits->numel() == model->meta().voc, "logits shape");
}

} // namespace

int main() {
    test_layer_prefill_and_decode();
    std::cout << "qwen2 layer NVIDIA forward test passed\n";
    return 0;
}
