#pragma once

#include "../../tensor/tensor.hpp"
#include "../core/model.hpp"
#include "../../sampling/sampling.hpp"

#include <memory>
#include <vector>

namespace llaisys {

class Qwen2;
using Qwen2_t = std::shared_ptr<Qwen2>;

using Qwen2Request = core::ModelRequest;
using qwen2_request_t = core::model_request_t;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

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

class Qwen2 : public core::Model {
protected:
    Qwen2Meta _meta;
    Qwen2Tensors _tensors;

    void bind_kv_caches() override;
    void bind_context_runtime(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    std::vector<int64_t> sample_from_logits(tensor_t logits, core::BatchPack &pack);

public:
    Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Qwen2() override;

    std::vector<int64_t> forward(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs);
    core::BatchPack prepare_prefill(const std::vector<seq_t> &seqs);
    core::BatchPack prepare_decode(const std::vector<seq_t> &seqs);

    tensor_t last_logits(size_t batch_size) const;
    const Qwen2Meta &meta() const { return this->_meta; }
};

} // namespace llaisys
