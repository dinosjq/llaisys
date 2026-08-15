#pragma once

#include "../../framework/model_context.hpp"

namespace llaisys::framework {

void run_llama_layer_stack(ModelContext &ctx, bool is_prefill);

} // namespace llaisys::framework
