#pragma once

#include "../../tensor/tensor.hpp"
#include "../meta/model_meta.hpp"
#include "../weight/attn_weights.hpp"
#include "../weight/ffn_weights.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

namespace llaisys::model {

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

// Base context: meta / runtime / KV / polymorphic weight slots.
class ModelContext {
public:
    ModelContext() = default;
    ModelContext(const ModelContext &) = delete;
    ModelContext &operator=(const ModelContext &) = delete;
    ModelContext(ModelContext &&) = default;
    ModelContext &operator=(ModelContext &&) = default;
    virtual ~ModelContext() = default;

    model_meta_t meta;
    BatchRuntime runtime{};
    std::vector<attn_weights_t> attns;
    std::vector<ffn_weights_t> ffns;
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
