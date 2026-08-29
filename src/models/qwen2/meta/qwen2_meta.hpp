#pragma once

#include "../../../model/meta/model_meta.hpp"
#include "../../../model/runtime/model_context.hpp"
#include "../../../utils/byte_map.hpp"

#include "llaisys.h"

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace llaisys::model {

class Qwen2Meta : public ModelMeta {
public:
    // info
    llaisysDataType_t dtype = LLAISYS_DTYPE_BF16;
    size_t nlayer = 0;
    size_t hs = 0;
    size_t nh = 0;
    size_t nkvh = 0;
    size_t dh = 0;
    size_t di = 0;
    size_t maxseq = 0;
    size_t voc = 0;
    float epsilon = 1e-6f;
    float theta = 10000.f;
    int64_t end_token = 0;
    float rope_scale = 1.f;         // Llama-3 factor <=1 表示不 scaling
    bool quantize = false;  // 统一 INT8 量化：权重 W8A8 + KV cache INT8（NVIDIA only）

    // transfer
    static Qwen2Meta cast(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv);

protected:
    void _load(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv);
};

using qwen2_meta_t = std::unique_ptr<Qwen2Meta>;

inline void Qwen2Meta::_load(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    using llaisys::utils::get_or_as;
    using llaisys::utils::get_or_f32;
    using llaisys::utils::get_or_string;
    using llaisys::utils::parse_torch_dtype;
    using llaisys::utils::require_as;

    this->dtype = parse_torch_dtype(get_or_string(kv, "torch_dtype", "bfloat16"));
    this->nlayer = static_cast<size_t>(require_as<int64_t>(kv, "num_hidden_layers"));
    this->hs = static_cast<size_t>(require_as<int64_t>(kv, "hidden_size"));
    this->nh = static_cast<size_t>(require_as<int64_t>(kv, "num_attention_heads"));
    this->nkvh = static_cast<size_t>(get_or_as<int64_t>(kv, "num_key_value_heads", static_cast<int64_t>(this->nh)));
    if (kv.count("head_dim")) {
        this->dh = static_cast<size_t>(require_as<int64_t>(kv, "head_dim"));
    } else {
        if (this->nh == 0) {
            throw std::invalid_argument("num_attention_heads is zero");
        }
        this->dh = this->hs / this->nh;
    }
    this->di = static_cast<size_t>(require_as<int64_t>(kv, "intermediate_size"));
    if (kv.count("max_position_embeddings")) {
        this->maxseq = static_cast<size_t>(require_as<int64_t>(kv, "max_position_embeddings"));
    } else if (kv.count("max_seq_len")) {
        this->maxseq = static_cast<size_t>(require_as<int64_t>(kv, "max_seq_len"));
    } else {
        this->maxseq = 0;
    }
    this->voc = static_cast<size_t>(require_as<int64_t>(kv, "vocab_size"));
    this->epsilon = get_or_f32(kv, "rms_norm_eps", 1e-6f);
    this->theta = get_or_f32(kv, "rope_theta", 10000.f);
    this->end_token = get_or_as<int64_t>(kv, "eos_token_id", 0);
    this->rope_scale = get_or_f32(kv, "rope_scaling.factor", 1.f);
    this->quantize = get_or_as<int64_t>(kv, "quantize", 0) != 0;
}

inline Qwen2Meta Qwen2Meta::cast(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) {
    auto meta = Qwen2Meta();
    meta._load(kv);
    return meta;
}
} // namespace llaisys::model
