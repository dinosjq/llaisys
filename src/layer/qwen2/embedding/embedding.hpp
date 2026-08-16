#pragma once

#include "../../layer.hpp"
#include "../../../models/qwen2/qwen2_context.hpp"

namespace llaisys::model {

class Qwen2Embedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);
};

} // namespace llaisys::model
