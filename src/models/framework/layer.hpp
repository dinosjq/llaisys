#pragma once

#include "model_context.hpp"

namespace llaisys::framework {

class Layer {
public:
    virtual ~Layer() = default;
    virtual void forward(ModelContext &ctx) = 0;
};

} // namespace llaisys::framework
