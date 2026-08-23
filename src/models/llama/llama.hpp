#pragma once

#include "../qwen2/qwen2.hpp"
#include "runtime/llama_context.hpp"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::model {

class Llama : public Qwen2 {
public:
    Llama(llaisysDeviceType_t device, int device_id);
    ~Llama() override = default;

    void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) override;
    void _build_layers();
    engine::BatchOutput forward(const engine::BatchInput &input) override;

    LlamaContext &llama_ctx();
};

} // namespace llaisys::model
