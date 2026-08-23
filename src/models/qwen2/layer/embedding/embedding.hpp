#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/qwen2_context.hpp"
#include "../../weight/qwen2_weights.hpp"

namespace llaisys::model {

class Qwen2Embedding : public Layer {
public:
    explicit Qwen2Embedding(Qwen2Weights &weights);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    Qwen2Weights &_w;
};

} // namespace llaisys::model
