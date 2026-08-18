#pragma once

#include <memory>

namespace llaisys::model {

// Empty polymorphic root. Concrete fields live on model-specific subclasses
// (e.g. Qwen2Meta). Same pattern as AttnWeights / FfnWeights.
class ModelMeta {
public:
    virtual ~ModelMeta() = default;
};
using model_meta_t = std::shared_ptr<ModelMeta>;

} // namespace llaisys::model
