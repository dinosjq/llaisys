#pragma once

#include "../layer.hpp"

#include <cstddef>

namespace llaisys::framework {

class Qwen2Embedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

class Qwen2Attention : public Layer {
public:
    explicit Qwen2Attention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *w_;
    size_t layer_;
};

class Qwen2FFN : public Layer {
public:
    explicit Qwen2FFN(const FfnWeights *weights, size_t layer_index_for_debug = 0);
    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *w_;
    size_t layer_index_for_debug_;
};

void run_qwen2_layer_stack(ModelContext &ctx, bool is_prefill);

} // namespace llaisys::framework
