#include "sampling.hpp"

#include "../utils.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace llaisys {

int clamp_top_k(int top_k, size_t candidate_count) {
    CHECK_ARGUMENT(top_k > 0, "top_k must be positive");
    return std::min({top_k, MAX_TOP_K, static_cast<int>(candidate_count)});
}

int64_t sample_token(std::vector<SamplingCandidate> candidates, int top_k, float top_p, float temperature,
                     std::mt19937 &rng) {
    CHECK_ARGUMENT(!candidates.empty(), "sampling requires at least one candidate");
    CHECK_ARGUMENT(top_p > 0.0f && top_p <= 1.0f, "top_p must be in (0, 1]");
    const int k = clamp_top_k(top_k, candidates.size());
    std::sort(candidates.begin(), candidates.end(),
              [](const SamplingCandidate &left, const SamplingCandidate &right) { return left.logit > right.logit; });
    if (temperature <= 0.0f || k == 1) {
        return candidates.front().token_id;
    }
    candidates.resize(k);

    const float max_logit = candidates.front().logit;
    std::vector<float> probabilities(candidates.size());
    float total = 0.0f;
    for (size_t i = 0; i < candidates.size(); ++i) {
        probabilities[i] = std::exp((candidates[i].logit - max_logit) / temperature);
        total += probabilities[i];
    }
    for (float &probability : probabilities) {
        probability /= total;
    }

    float cumulative = 0.0f;
    size_t retained = 0;
    do {
        cumulative += probabilities[retained++];
    } while (retained < probabilities.size() && cumulative < top_p);
    probabilities.resize(retained);
    candidates.resize(retained);
    const float retained_total = std::accumulate(probabilities.begin(), probabilities.end(), 0.0f);
    for (float &probability : probabilities) {
        probability /= retained_total;
    }

    std::discrete_distribution<size_t> distribution(probabilities.begin(), probabilities.end());
    return candidates[distribution(rng)].token_id;
}

} // namespace llaisys
