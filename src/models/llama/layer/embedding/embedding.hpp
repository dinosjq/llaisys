#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"

namespace llaisys::model {

class LlamaEmbedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);
};

} // namespace llaisys::model
