#pragma once

#include "../../../../model/layer/layer.hpp"
#include "../../runtime/qwen2_context.hpp"
#include "../../weight/qwen2_weights.hpp"

namespace llaisys::model {

// 输出层：gather 每序列最后一个 token 的 hidden state → rms_norm → linear(lm_head)
class Qwen2Logit : public Layer {
public:
    explicit Qwen2Logit(Qwen2Weights &weights);
    void forward(ModelContext &ctx) override;
    void forward(Qwen2Context &ctx);

private:
    Qwen2Weights &_w;
};

} // namespace llaisys::model
