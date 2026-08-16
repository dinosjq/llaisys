#pragma once

#include "../model/model_context.hpp"

namespace llaisys::model {

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(ModelContext &ctx) = 0;
};

} // namespace llaisys::model
