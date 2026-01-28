#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, const float &eps, const size_t &n, const size_t &m) {
    for (size_t i = 0; i < n; ++ i){
        float den = 0.0f;
        for (size_t j = 0; j < m; ++j) {
            const size_t k = i * m + j;
            float t = 0.0f;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                t = llaisys::utils::cast<float>(in[k]);
            } else {
                t = in[k];
            }
            den += t * t;
        }
        den /= m;
        den += eps;
        den = std::sqrt(den);
        for (size_t j = 0; j < m; ++ j){
            const size_t k = i * m + j;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[k] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(weight[j]) * llaisys::utils::cast<float>(in[k]) / den);
            } else {
                out[k] = weight[j] * in[k] / den;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, const float &eps, llaisysDataType_t dtype,
              const size_t &n, const size_t &m) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                        reinterpret_cast<const float *>(weight), eps, n, m);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const llaisys::bf16_t *>(weight), eps, n, m);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const llaisys::fp16_t *>(weight), eps, n, m);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
