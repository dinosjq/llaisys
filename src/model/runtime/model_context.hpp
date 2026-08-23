#pragma once

#include <memory>

namespace llaisys::model {

class ModelContext {
public:
    ModelContext() = default;
    virtual ~ModelContext() = default;
};

using model_context_t = std::unique_ptr<ModelContext>;

} // namespace llaisys::model
