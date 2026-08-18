#pragma once

#include <memory>

namespace llaisys::model {

// Empty polymorphic root. Concrete fields live on model-specific subclasses
// (e.g. QwenAttnWeights). Later W8 variants inherit those and add scales.
class AttnWeights {
public:
    virtual ~AttnWeights() = default;
};
using attn_weights_t = std::shared_ptr<AttnWeights>;

} // namespace llaisys::model
