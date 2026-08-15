#pragma once

#include "../qwen2/qwen2.hpp"

namespace llaisys {

class Llama : public Qwen2 {
public:
    using Qwen2::Qwen2;
    ~Llama() override = default;

    std::vector<int64_t> forward(core::BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) override;
};

} // namespace llaisys
