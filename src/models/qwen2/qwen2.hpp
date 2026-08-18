#pragma once

#include "../../tensor/tensor.hpp"
#include "../../model/model.hpp"
#include "runtime/qwen2_context.hpp"
#include "meta/qwen2_meta.hpp"
#include "../../sampling/sampling.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys {

class Qwen2;
using Qwen2_t = std::shared_ptr<Qwen2>;

using Qwen2Request = model::ModelRequest;
using qwen2_request_t = model::model_request_t;
using model::Qwen2Meta;

class Qwen2 : public model::Model {
protected:
    void bind_kv_caches() override;
    void bind_context_runtime(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;
    std::vector<int64_t> sample_from_logits(tensor_t logits, model::BatchPack &pack);
    void allocate_runtime();

public:
    // Device-only construction; call set_meta before set_weight / submit.
    Qwen2(llaisysDeviceType_t device, int *device_ids, int ndevice,
          std::unique_ptr<model::Qwen2Context> ctx = nullptr);
    ~Qwen2() override;

    void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) override;
    void set_weight(const std::string &hf_name, tensor_t tensor) override;

    std::vector<int64_t> forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    tensor_t last_logits(size_t batch_size) const;
    const Qwen2Meta &meta() const { return model::qwen_meta(*this->_ctx); }
    Qwen2Meta &meta() { return model::qwen_meta(*this->_ctx); }
};

} // namespace llaisys
