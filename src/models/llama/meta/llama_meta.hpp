#pragma once

#include "../../qwen2/meta/qwen2_meta.hpp"
#include "../../../model/layer/utils/rope_inv_freq.hpp"
#include "../../../utils/byte_map.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::model {

class LlamaMeta : public Qwen2Meta {
public:
    // Llama-3 rope_scaling 中几乎不变的部分（变的是 rope_scale）
    static constexpr float kRopeLowFreqFactor = 1.f;
    static constexpr float kRopeHighFreqFactor = 4.f;
    static constexpr int64_t kRopeOriginalMaxPos = 8192;
    static constexpr size_t kMaxSeqCap = 8192;

    // Llama defaults → Qwen2Meta::load_from_map → clamp maxseq.
    static std::shared_ptr<LlamaMeta> from_map(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv);

    // Host inv_freq[dh/2]，含可选 llama3 rope scaling。
    std::vector<float> make_inv_freq() const;
};
using llama_meta_t = std::shared_ptr<LlamaMeta>;

inline llama_meta_t LlamaMeta::from_map(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    using llaisys::utils::get_or_string;
    using llaisys::utils::put_as;

    llaisys::utils::ByteMap m = kv;
    if (!m.count("rms_norm_eps")) {
        put_as<float>(m, "rms_norm_eps", 1e-5f);
    }
    if (!m.count("rope_theta")) {
        put_as<float>(m, "rope_theta", 500000.f);
    }
    if (!m.count("max_position_embeddings") && !m.count("max_seq_len")) {
        put_as<int64_t>(m, "max_position_embeddings", static_cast<int64_t>(kMaxSeqCap));
    }

    const std::string rope_type =
        get_or_string(m, "rope_scaling.rope_type", get_or_string(m, "rope_scaling.type", ""));
    if (rope_type == "llama3") {
        if (!m.count("rope_scaling.factor")) {
            put_as<float>(m, "rope_scaling.factor", 1.f);
        }
    } else {
        put_as<float>(m, "rope_scaling.factor", 1.f);
    }

    auto meta = std::make_shared<LlamaMeta>();
    meta->load_from_map(m);
    if (meta->maxseq > kMaxSeqCap) {
        meta->maxseq = kMaxSeqCap;
    }
    return meta;
}

inline std::vector<float> LlamaMeta::make_inv_freq() const {
    auto host_inv = layers::utils::make_rope_inv_freq(this->dh, this->theta);
    if (this->rope_scale > 1.f) {
        host_inv = layers::utils::apply_llama3_rope_scaling(std::move(host_inv), this->rope_scale, kRopeLowFreqFactor,
                                                            kRopeHighFreqFactor, kRopeOriginalMaxPos);
    }
    return host_inv;
}

inline LlamaMeta &llama_meta(ModelContext &ctx) {
    return *std::static_pointer_cast<LlamaMeta>(ctx.meta);
}

inline const LlamaMeta &llama_meta(const ModelContext &ctx) {
    return *std::static_pointer_cast<const LlamaMeta>(ctx.meta);
}

} // namespace llaisys::model
