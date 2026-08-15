#pragma once

#include "model_context.hpp"
#include "weight_role.hpp"

namespace llaisys::framework {

void set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor);

} // namespace llaisys::framework
