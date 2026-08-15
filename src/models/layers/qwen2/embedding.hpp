#pragma once

#include "../../framework/layer.hpp"

namespace llaisys::framework {

class Qwen2Embedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
};

} // namespace llaisys::framework
