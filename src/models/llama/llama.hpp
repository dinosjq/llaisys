#pragma once

#include "../qwen2/qwen2.hpp"
#include "llama_context.hpp"

namespace llaisys {

class Llama : public Qwen2 {
public:
    Llama(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Llama() override = default;

    std::vector<int64_t> forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    model::LlamaContext &llama_ctx();
};

} // namespace llaisys
