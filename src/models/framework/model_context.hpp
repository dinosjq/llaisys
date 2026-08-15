#pragma once

#include "../../tensor/tensor.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::framework {

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

struct BatchRuntime {
    std::vector<int64_t> token_ids, pos_ids, cut_idx, tot_len;
    size_t max_seq_len;
    bool is_prefill;
};

struct WorkspaceTensors {
    tensor_t x;
    tensor_t x_norm;
    tensor_t q;
    tensor_t k;
    tensor_t v;
    tensor_t q_rope;
    tensor_t k_rope;
    tensor_t attn_val;
    tensor_t attn_out;
    tensor_t x_attn;
    tensor_t m_norm;
    tensor_t gate;
    tensor_t up;
    tensor_t swiglu;
    tensor_t down;
    tensor_t x_mlp;
    tensor_t block_ids;
    tensor_t cut_idx;
    tensor_t tot_len;
    tensor_t pos_ids;
    tensor_t x_part;
    tensor_t x_part_norm;
    tensor_t logits;
    tensor_t capture_post_attn0;
    tensor_t capture_post_ffn0;
};

struct ModelContext {
    ModelMeta meta;
    BatchRuntime runtime;
    WorkspaceTensors workspace;
    std::vector<AttnWeights> attns;
    std::vector<FfnWeights> ffns;
    std::vector<tensor_t> k_caches;
    std::vector<tensor_t> v_caches;
    tensor_t in_embed, out_embed, out_norm_w;
};

} // namespace llaisys::framework
