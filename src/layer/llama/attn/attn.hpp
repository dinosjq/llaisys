#pragma once

#include "../../layer.hpp"
#include "../../../models/llama/llama_context.hpp"

#include <cstddef>

namespace llaisys::model {

class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    const AttnWeights *_w;
    size_t _layer;
};

} // namespace llaisys::model
