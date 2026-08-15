#pragma once

#include "../../framework/layer.hpp"

#include <cstddef>

namespace llaisys::framework {

class Qwen2Attention : public Layer {
public:
    explicit Qwen2Attention(const AttnWeights *weights, size_t layer = 0);

    void forward(ModelContext &ctx) override;

private:
    const AttnWeights *w_;
    size_t layer_;
};

} // namespace llaisys::framework
