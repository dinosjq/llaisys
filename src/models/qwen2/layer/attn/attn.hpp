#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/qwen2_context.hpp"
#include "../../weight/qwen2_weights.hpp"

#include <cstddef>

namespace llaisys::model {

class Qwen2Attention : public Layer {
public:
    explicit Qwen2Attention(qwen_attn_weights_t weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    qwen_attn_weights_t _w;
    size_t _layer;
};

} // namespace llaisys::model
