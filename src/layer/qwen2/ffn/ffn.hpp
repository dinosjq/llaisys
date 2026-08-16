#pragma once

#include "../../layer.hpp"
#include "../../../models/qwen2/qwen2_context.hpp"

namespace llaisys::model {

class Qwen2FFN : public Layer {
public:
    explicit Qwen2FFN(const FfnWeights *weights);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    const FfnWeights *_w;
};

} // namespace llaisys::model
