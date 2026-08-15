#pragma once

#include "../models/core/model_context.hpp"

namespace llaisys::core {

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(ModelContext &ctx) = 0;
};

} // namespace llaisys::core
