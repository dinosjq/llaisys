#pragma once

#include "../../layer.hpp"
#include "../../../models/qwen2/qwen2_context.hpp"

#include <cstddef>

namespace llaisys::model {

class Qwen2Attention : public Layer {
public:
    explicit Qwen2Attention(const AttnWeights *weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    const AttnWeights *_w;
    size_t _layer;
};

} // namespace llaisys::model
