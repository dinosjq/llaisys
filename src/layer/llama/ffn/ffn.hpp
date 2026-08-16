#pragma once

#include "../../layer.hpp"
#include "../../../models/llama/llama_context.hpp"

namespace llaisys::model {

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(const FfnWeights *weights);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    const FfnWeights *_w;
};

} // namespace llaisys::model
