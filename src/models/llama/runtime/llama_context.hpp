#pragma once

#include "../../qwen2/runtime/qwen2_context.hpp"

namespace llaisys::model {

class LlamaContext : public Qwen2Context {
public:
    // 预计算 RoPE inv_freq[dh/2]（float），供 rope(inv_freq) 入口使用
    tensor_t inv_freq;
};

} // namespace llaisys::model
