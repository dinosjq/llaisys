#pragma once

#include <cstdint>
#include <random>
#include <vector>

namespace llaisys {

constexpr int MAX_TOP_K = 100;

struct SamplingCandidate {
    float logit;
    int64_t token_id;
};

int clamp_top_k(int top_k, size_t candidate_count);
int64_t sample_token(std::vector<SamplingCandidate> candidates, int top_k, float top_p, float temperature,
                     std::mt19937 &rng);

} // namespace llaisys
