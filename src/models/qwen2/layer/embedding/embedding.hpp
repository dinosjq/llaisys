#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/qwen2_context.hpp"

namespace llaisys::model {

class Qwen2Embedding : public Layer {
public:
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);
};

} // namespace llaisys::model
