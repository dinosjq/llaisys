#pragma once

#include "../../framework/layer.hpp"

#include <cstddef>

namespace llaisys::framework {

// Llama attention: same GQA + RoPE + paged path as Qwen2; biases stay null.
class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(const AttnWeights *weights, size_t layer = 0);

    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *w_;
    size_t layer_;
};

} // namespace llaisys::framework
