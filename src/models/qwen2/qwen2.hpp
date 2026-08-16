#pragma once

#include "../../tensor/tensor.hpp"
#include "../../model/model.hpp"
#include "qwen2_context.hpp"
#include "../../sampling/sampling.hpp"

#include <memory>
#include <vector>

namespace llaisys {

class Qwen2;
using Qwen2_t = std::shared_ptr<Qwen2>;

using Qwen2Request = model::ModelRequest;
using qwen2_request_t = model::model_request_t;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
    float rope_scale = 1.f; // Llama-3 factor；<=1 表示不 scaling
};

class Qwen2 : public model::Model {
protected:
    Qwen2Meta _meta;

    void bind_kv_caches() override;
    void bind_context_runtime(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    std::vector<int64_t> sample_from_logits(tensor_t logits, model::BatchPack &pack);

public:
    // ctx 为空时默认构造 Qwen2Context；Llama 传入 LlamaContext
    Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice,
          std::unique_ptr<model::Qwen2Context> ctx = nullptr);
    ~Qwen2() override;

    std::vector<int64_t> forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    tensor_t last_logits(size_t batch_size) const;
    const Qwen2Meta &meta() const { return this->_meta; }
};

} // namespace llaisys
