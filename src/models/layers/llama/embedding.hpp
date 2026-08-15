#pragma once

#include "../../framework/layer.hpp"

namespace llaisys::framework {

class LlamaEmbedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

} // namespace llaisys::framework
