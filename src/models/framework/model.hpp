#pragma once

#include "model_context.hpp"
#include "weight_role.hpp"

#include <cstddef>

namespace llaisys::framework {

class Model {
public:
    virtual ~Model() = default;

    // Upload-boundary helper: map a public WeightRole into the layered ModelContext slots.
    static void set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor);

    // TODO: lift prepare_block_table / prepare_prefill / prepare_decode from Qwen2 into Model.
    // These stubs reserve the Model API surface for the later production migration.
    // void prepare_block_table(...);
    // void prepare_prefill(...);
    // void prepare_decode(...);
};

} // namespace llaisys::framework
