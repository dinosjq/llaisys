#pragma once

#include <memory>

namespace llaisys::model {

// Empty polymorphic root. Concrete fields live on model-specific subclasses
// (e.g. QwenFfnWeights). Later W8 variants inherit those and add scales.
class FfnWeights {
public:
    virtual ~FfnWeights() = default;
};
using ffn_weights_t = std::shared_ptr<FfnWeights>;

} // namespace llaisys::model
