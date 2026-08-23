#pragma once

#include <memory>

namespace llaisys::model {

class ModelWeights {
public:
    virtual ~ModelWeights() = default;
};
using model_weights_t = std::unique_ptr<ModelWeights>;

} // namespace llaisys::model
