#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/llama_context.hpp"
#include "../../weight/llama_weights.hpp"

namespace llaisys::model {

class LlamaFFN : public Layer {
public:
    explicit LlamaFFN(llama_ffn_weights_t weights);
    void forward(ModelContext &ctx) override;
    void forward(LlamaContext &ctx);

private:
    llama_ffn_weights_t _w;
};

} // namespace llaisys::model
