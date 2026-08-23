#pragma once

#include <memory>

namespace llaisys::model {

class ModelMeta {
public:
    virtual ~ModelMeta() = default;
};

using model_meta_t = std::unique_ptr<ModelMeta>;
} // namespace llaisys::model
