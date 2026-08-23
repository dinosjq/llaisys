#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/qwen2_context.hpp"
#include "../../weight/qwen2_weights.hpp"

namespace llaisys::model {

class Qwen2FFN : public Layer {
public:
    explicit Qwen2FFN(QwenFfnWeights &weights);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    QwenFfnWeights &_w;
};

} // namespace llaisys::model
