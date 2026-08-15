#pragma once

#include "../layer.hpp"

#include <cstddef>

namespace llaisys::framework {

class LlamaEmbedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *w_;
    size_t layer_;
};

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(const FfnWeights *weights, size_t layer_index_for_debug = 0);
    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *w_;
    size_t layer_index_for_debug_;
};

void run_llama_layer_stack(ModelContext &ctx, bool is_prefill);

} // namespace llaisys::framework
