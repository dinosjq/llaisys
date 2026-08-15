#pragma once

#include "../layer.hpp"

#include <cstddef>

namespace llaisys::core {

class LlamaEmbedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *_w;
    size_t _layer;
};

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(const FfnWeights *weights);
    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *_w;
};

} // namespace llaisys::core
