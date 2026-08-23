#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"
#include "../../../qwen2/weight/qwen2_weights.hpp"

namespace llaisys::model {

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(QwenFfnWeights &weights);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    QwenFfnWeights &_w;
};

} // namespace llaisys::model
