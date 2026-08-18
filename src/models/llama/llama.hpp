#pragma once

#include "../qwen2/qwen2.hpp"
#include "runtime/llama_context.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys {

class Llama : public Qwen2 {
public:
    Llama(llaisysDeviceType_t device, int *device_ids, int ndevice);
    ~Llama() override = default;

    void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) override;
    std::vector<int64_t> forward(model::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;

    model::LlamaContext &llama_ctx();
};

} // namespace llaisys
