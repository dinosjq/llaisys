#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

namespace llaisys {

struct Qwen2Pack {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx;
    std::vector<int64_t> tot_len;
    std::vector<int> top_k;
    std::vector<float> top_p;       // per-seq 采样参数
    std::vector<float> temperature; // per-seq 采样参数
    std::vector<std::mt19937 *> rngs;
    size_t max_seq_len;
};

} // namespace llaisys
