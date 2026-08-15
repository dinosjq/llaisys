#pragma once

#include "../layer.hpp"

#include <cstddef>

namespace llaisys::core {

class Qwen2Embedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

class Qwen2Attention : public Layer {
public:
    explicit Qwen2Attention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *_w;
    size_t _layer;
};

class Qwen2FFN : public Layer {
public:
    explicit Qwen2FFN(const FfnWeights *weights);
    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *_w;
};

} // namespace llaisys::core
