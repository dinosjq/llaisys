#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace llaisys::layers::utils {

inline constexpr float kRopePi = 3.14159265358979323846f;

// 标准 RoPE：inv_freq[j] = 1 / theta^(2j/d)
inline std::vector<float> make_rope_inv_freq(size_t dh, float theta) {
    const size_t half = dh / 2;
    std::vector<float> inv_freq(half);
    for (size_t j = 0; j < half; ++j) {
        inv_freq[j] = 1.0f / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(dh));
    }
    return inv_freq;
}

// Llama-3 rope_scaling（与 transformers ROPE_INIT_FUNCTIONS['llama3'] 对齐）
inline std::vector<float> apply_llama3_rope_scaling(std::vector<float> inv_freq, float factor, float low_freq_factor,
                                                    float high_freq_factor, int64_t old_context_len) {
    const float low_freq_wavelen = static_cast<float>(old_context_len) / low_freq_factor;
    const float high_freq_wavelen = static_cast<float>(old_context_len) / high_freq_factor;
    const float two_pi = 2.0f * kRopePi;

    for (float &freq : inv_freq) {
        const float wavelen = two_pi / freq;
        float freq_llama = (wavelen > low_freq_wavelen) ? (freq / factor) : freq;
        if (!(wavelen < high_freq_wavelen) && !(wavelen > low_freq_wavelen)) {
            const float smooth =
                (static_cast<float>(old_context_len) / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor);
            freq_llama = (1.0f - smooth) * freq_llama / factor + smooth * freq_llama;
        }
        freq = freq_llama;
    }
    return inv_freq;
}

} // namespace llaisys::layers::utils
