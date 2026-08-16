#pragma once

#include "../../layer.hpp"
#include "../../../models/llama/llama_context.hpp"

namespace llaisys::model {

class LlamaEmbedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);
};

} // namespace llaisys::model
