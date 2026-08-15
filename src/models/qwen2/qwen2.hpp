#pragma once

#include "../../../include/llaisys/models/qwen2.h"
#include "../../tensor/tensor.hpp"
#include "../core/model.hpp"
#include "../../sampling/sampling.hpp"

#include <memory>
#include <vector>

namespace llaisys {

class Qwen2;
using Qwen2_t = std::shared_ptr<Qwen2>;

using Qwen2Request = framework::ModelRequest;
using qwen2_request_t = framework::model_request_t;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

using Qwen2Weights = ::LlaisysQwen2Weights;

struct Qwen2Tensors {
    tensor_t _x_norm;
    tensor_t _q;
    tensor_t _k;
    tensor_t _v;
    tensor_t _q_rope;
    tensor_t _k_rope;
    tensor_t _attn_val;
    tensor_t _attn_out;
    tensor_t _x_attn;
    tensor_t _m_norm;
    tensor_t _gate;
    tensor_t _up;
    tensor_t _swiglu;
    tensor_t _down;
    tensor_t _x_mlp;
    tensor_t _dev_block_ids;
    tensor_t _dev_cut_idx;
    tensor_t _dev_tot_len;
    tensor_t _dev_token_ids;
    tensor_t _dev_pos_ids;
    tensor_t _x;
    tensor_t _x_part;
    tensor_t _x_part_norm;
    tensor_t _logits;
    tensor_t _top_idx;
    tensor_t _top_val;
    tensor_t _attn_acc;
    tensor_t _attn_sum;
    tensor_t _attn_max;
};

class Qwen2 : public framework::Model {
private:
    Qwen2Meta _meta;
    Qwen2Weights _weights;
    Qwen2Tensors _tensors;
    bool use_llama_stack_ = false; // LLAISYS_LAYER_STACK=llama

    void sync_weights() override;
    void bind_context_runtime(framework::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    std::vector<int64_t> sample_from_logits(tensor_t logits, framework::BatchPack &pack);

public:
    Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2() override;

    // Layer 组装前向（供测试与诊断直接调用）
    std::vector<int64_t> forward(framework::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    // 块表 / prepare（委托基类）
    std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs);
    framework::BatchPack prepare_prefill(const std::vector<seq_t> &seqs);
    framework::BatchPack prepare_decode(const std::vector<seq_t> &seqs);

    tensor_t last_logits(size_t batch_size) const;
    Qwen2Weights &weights() { return this->_weights; }
    const Qwen2Meta &meta() const { return this->_meta; }
};

} // namespace llaisys
