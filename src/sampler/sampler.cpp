#include "sampler.hpp"

#include "llaisys/runtime.h"
#include "../ops/top_k/op.hpp"
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

std::vector<int64_t> Sampler::sample(tensor_t logits, const engine::BatchPack &pack,
                                     llaisysDeviceType_t device, int device_id,
                                     llaisysDataType_t dtype) const {
    const size_t batch_size = pack.batch_size;

    // topk：取每个序列 logits 的前 K 个候选（K = 所有序列 top_k 的最大值）
    const size_t K = static_cast<size_t>(*std::max_element(pack.top_ks.begin(), pack.top_ks.end()));
    tensor_t top_idx = Tensor::create({batch_size, K}, LLAISYS_DTYPE_I64, device, device_id);
    tensor_t top_val = Tensor::create({batch_size, K}, dtype, device, device_id);
    ops::topk(top_idx, top_val, logits, K);
    // 非 CPU 设备：把 topk 结果拷回 CPU
    if (device != LLAISYS_DEVICE_CPU) {
        top_idx = top_idx->to(LLAISYS_DEVICE_CPU, 0);
        top_val = top_val->to(LLAISYS_DEVICE_CPU, 0);
    }
    const std::vector<int64_t> all_idx = top_idx->reshape({batch_size * K})->to_vector<int64_t>();
    const std::vector<float> all_val = top_val->reshape({batch_size * K})->to_vector<float>();

    // 逐序列按各自采样参数选 token
    std::vector<int64_t> result(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
        std::vector<SamplingCandidate> candidates;
        candidates.reserve(K);
        for (size_t j = 0; j < K; ++j) {
            const size_t offset = i * K + j;
            candidates.push_back({all_val[offset], all_idx[offset]});
        }
        result[i] = sample_token(candidates, pack.top_ks[i], pack.top_ps[i], pack.temperatures[i], *pack.rngs[i]);
    }
    return result;
}

} // namespace llaisys
