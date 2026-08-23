#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"
#include "../../../qwen2/weight/qwen2_weights.hpp"

#include <cstddef>

namespace llaisys::model {

class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(QwenAttnWeights &weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    QwenAttnWeights &_w;
    size_t _layer;
};

} // namespace llaisys::model
