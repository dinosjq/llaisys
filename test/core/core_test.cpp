#include "../../src/kv_cache/block_manager.hpp"
#include "../../src/models/sampling.hpp"
#include "../../src/scheduler/scheduler.hpp"

#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>

namespace {

void require(bool condition, const char *message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void test_prefix_cache_requires_elementwise_block_match() {
    constexpr size_t block_tokens = 64;
    llaisys::BlockManager manager(block_tokens, 4);
    std::vector<int64_t> first_tokens(block_tokens * 2);
    for (size_t i = 0; i < first_tokens.size(); ++i) {
        first_tokens[i] = static_cast<int64_t>(i);
    }
    auto first = std::make_shared<llaisys::Sequence>(first_tokens.data(), first_tokens.size());
    require(manager.can_allocate(first) == 0, "fresh sequence unexpectedly reused a prefix");
    manager.allocate(first, 0);
    first->scheduled_token_num() = block_tokens;
    manager.hash_blocks(first);
    first->cached_token_num() = block_tokens;

    auto equal = std::make_shared<llaisys::Sequence>(first_tokens.data(), first_tokens.size());
    require(manager.can_allocate(equal) == 1, "equal complete cache block was not reused");

    auto mismatch_tokens = first_tokens;
    mismatch_tokens[2] = 999;
    auto mismatch = std::make_shared<llaisys::Sequence>(mismatch_tokens.data(), mismatch_tokens.size());
    require(manager.can_allocate(mismatch) == 0, "mismatch outside index one reused a cache block");
}

void test_sampling_policy() {
    std::vector<llaisys::SamplingCandidate> candidates{{1.0f, 10}, {2.0f, 20}, {0.0f, 30}};
    std::mt19937 greedy_rng(1);
    require(llaisys::sample_token(candidates, 3, 0.9f, 0.0f, greedy_rng) == 20,
            "non-positive temperature must select the maximum logit");
    require(llaisys::sample_token(candidates, 1, 0.1f, 1.0f, greedy_rng) == 20,
            "top_k one must select the maximum logit");

    std::mt19937 crossing_rng(2);
    require(llaisys::sample_token({{0.0f, 1}, {0.0f, 2}}, 2, 0.5f, 1.0f, crossing_rng) == 1,
            "nucleus must retain the first candidate crossing top_p");

    std::mt19937 weighted_rng(3);
    const auto weighted = llaisys::sample_token({{8.0f, 7}, {0.0f, 8}}, 2, 1.0f, 1.0f, weighted_rng);
    require(weighted == 7, "sampling must use probability weights, not uniform candidate selection");
}

void test_chunked_prefill_makes_progress_without_generating_early() {
    llaisys::Scheduler scheduler(64, 4, 2, 1, 999);
    std::vector<int64_t> prompt{1, 2, 3, 4, 5};
    auto sequence = std::make_shared<llaisys::Sequence>(prompt.data(), prompt.size());
    scheduler.add(sequence);

    for (size_t expected_cached : {size_t(2), size_t(4)}) {
        auto [scheduled, is_prefill] = scheduler.schedule();
        require(is_prefill && scheduled.size() == 1, "chunk must schedule the queue front once");
        require(scheduled.front()->scheduled_token_num() <= 2, "scheduled tokens exceed remaining batch capacity");
        scheduler.postprocess(scheduled, {77}, true);
        require(sequence->cached_token_num() == expected_cached, "cached token count did not progress monotonically");
        require(sequence->token_num() == prompt.size(), "prefill generated a token before prompt completion");
    }

    auto [scheduled, is_prefill] = scheduler.schedule();
    require(is_prefill && scheduled.size() == 1, "final prompt chunk was not scheduled");
    scheduler.postprocess(scheduled, {77}, true);
    require(sequence->cached_token_num() == prompt.size(), "final prompt chunk did not finish caching");
    require(sequence->token_num() == prompt.size() + 1, "prefill completion must add exactly one generated token");
}

} // namespace

int main() {
    test_prefix_cache_requires_elementwise_block_match();
    test_sampling_policy();
    test_chunked_prefill_makes_progress_without_generating_early();
    std::cout << "llaisys core test smoke assertion passed\n";
    return 0;
}
