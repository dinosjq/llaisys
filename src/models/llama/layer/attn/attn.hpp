#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"
#include "../../weight/llama_weights.hpp"

#include <cstddef>

namespace llaisys::model {

class LlamaAttention : public Layer {
public:
    explicit LlamaAttention(llama_attn_weights_t weights, size_t layer = 0);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    llama_attn_weights_t _w;
    size_t _layer;
};

} // namespace llaisys::model
