#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"
#include "../../../qwen2/weight/qwen2_weights.hpp"

namespace llaisys::model {

class LlamaEmbedding : public Layer {
public:
    explicit LlamaEmbedding(Qwen2Weights &weights);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    Qwen2Weights &_w;
};

} // namespace llaisys::model
