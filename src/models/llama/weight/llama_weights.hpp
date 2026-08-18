#pragma once

#include "../../qwen2/weight/qwen2_weights.hpp"

namespace llaisys::model {

using LlamaAttnWeights = QwenAttnWeights;
using LlamaFfnWeights = QwenFfnWeights;
using llama_attn_weights_t = qwen_attn_weights_t;
using llama_ffn_weights_t = qwen_ffn_weights_t;

class LlamaWeights {
public:
    static llama_attn_weights_t attn(ModelContext &ctx, size_t layer) { 
        return Qwen2Weights::attn(ctx, layer); 
    }
    static llama_ffn_weights_t ffn(ModelContext &ctx, size_t layer) { 
        return Qwen2Weights::ffn(ctx, layer); 
    }
    static void init(ModelContext &ctx) { 
        Qwen2Weights::init(ctx); 
    }
    static bool set_weight(ModelContext &ctx, const std::string &hf_name, tensor_t tensor) {
        return Qwen2Weights::set_weight(ctx, hf_name, std::move(tensor));
    }
};

} // namespace llaisys::model
