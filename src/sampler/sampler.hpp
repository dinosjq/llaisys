#pragma once

#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include "../tensor/tensor.hpp"
#include "../engine/engine.hpp"

namespace llaisys {

constexpr int MAX_TOP_K = 100;

// 单个候选（logits 值 + token id）
struct SamplingCandidate {
    float logit;
    int64_t token_id;
};

// 钳制 top_k 到 [1, min(MAX_TOP_K, candidate_count)]
int clamp_top_k(int top_k, size_t candidate_count);

// 从候选列表按 top_k/top_p/temperature 采样一个 token
int64_t sample_token(std::vector<SamplingCandidate> candidates, int top_k, float top_p, float temperature,
                     std::mt19937 &rng);

// 从模型 logits 采样下一 token
// 内部：GPU/CPU topk → 拷回 CPU → 逐序列按采样参数选 token
class Sampler {
public:
    std::vector<int64_t> sample(tensor_t logits, const engine::BatchPack &pack,
                                llaisysDeviceType_t device, int device_id,
                                llaisysDataType_t dtype) const;
};

} // namespace llaisys
