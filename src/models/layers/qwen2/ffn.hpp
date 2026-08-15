#pragma once

#include "../../framework/layer.hpp"

#include <cstddef>

namespace llaisys::framework {

class Qwen2FFN : public Layer {
public:
    explicit Qwen2FFN(const FfnWeights *weights, size_t layer_index_for_debug = 0);

    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *w_;
    size_t layer_index_for_debug_;
};

} // namespace llaisys::framework
