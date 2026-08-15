#pragma once

#include "model_context.hpp"
#include "weight_role.hpp"
#include "../qwen2_pack.hpp"
#include "../../sequence/sequence.hpp"

#include <cstddef>
#include <vector>

namespace llaisys::framework {

class Model {
public:
    virtual ~Model() = default;

    // Upload-boundary helper: map a public WeightRole into the layered ModelContext slots.
    static void set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor);

    // Shared batch prepare helpers (lifted from Qwen2; return Qwen2Pack this migration task).
    static std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs, size_t block_table_width);
    static Qwen2Pack prepare_prefill(const std::vector<seq_t> &seqs);
    static Qwen2Pack prepare_decode(const std::vector<seq_t> &seqs);
};

} // namespace llaisys::framework
