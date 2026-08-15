#pragma once

#include "../../framework/layer.hpp"

#include <cstddef>

namespace llaisys::framework {

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(const FfnWeights *weights, size_t layer_index_for_debug = 0);

    void forward(ModelContext &ctx) override;

private:
    const FfnWeights *w_;
    size_t layer_index_for_debug_;
};

} // namespace llaisys::framework
