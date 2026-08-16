#pragma once

#include "../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace llaisys::model {

enum class WeightRole : int {
    InEmbed,
    OutEmbed,
    OutNorm,
    AttnNorm,
    AttnQ_W,
    AttnQ_B,
    AttnK_W,
    AttnK_B,
    AttnV_W,
    AttnV_B,
    AttnO_W,
    MlpNorm,
    MlpGate_W,
    MlpUp_W,
    MlpDown_W,
};

struct AttnWeights {
    tensor_t norm_w, q_w, q_b, k_w, k_b, v_w, v_b, o_w;
};

struct FfnWeights {
    tensor_t norm_w, gate_w, up_w, down_w;
};

struct ModelMeta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct BatchPack {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx;
    std::vector<int64_t> tot_len;
    std::vector<int> top_k;
    std::vector<float> top_p;
    std::vector<float> temperature;
    std::vector<std::mt19937 *> rngs;
    size_t max_seq_len;
};

struct BatchRuntime {
    std::vector<int64_t> token_ids, pos_ids, cut_idx, tot_len;
    size_t max_seq_len = 0;
    size_t tot_block_num = 0;
    bool is_prefill = true;
};

// 基类：权重 / meta / runtime / KV
class ModelContext {
public:
    virtual ~ModelContext() = default;

    ModelMeta meta{};
    BatchRuntime runtime{};
    std::vector<AttnWeights> attns;
    std::vector<FfnWeights> ffns;
    std::vector<tensor_t> k_caches;
    std::vector<tensor_t> v_caches;
    tensor_t in_embed, out_embed, out_norm_w;

    void resize_layers(size_t nlayer) {
        attns.resize(nlayer);
        ffns.resize(nlayer);
        k_caches.resize(nlayer);
        v_caches.resize(nlayer);
    }
};

} // namespace llaisys::model
