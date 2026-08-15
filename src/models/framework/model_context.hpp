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
};

struct WorkspaceTensors {
    tensor_t x;
    tensor_t x_norm;
    tensor_t m_norm;
    tensor_t gate;
    tensor_t up;
    tensor_t swiglu;
    tensor_t down;
    tensor_t x_mlp;
};

struct ModelContext {
    ModelMeta meta;
    BatchRuntime runtime;
    WorkspaceTensors workspace;
    std::vector<AttnWeights> attns;
    std::vector<FfnWeights> ffns;
    tensor_t in_embed, out_embed, out_norm_w;
};

} // namespace llaisys::framework
