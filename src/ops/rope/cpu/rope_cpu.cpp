#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, const float &theta, const size_t &seqlen, const size_t &head, const size_t &d) {
    const size_t stride_0 = head * d;
    const size_t stride_1 = d;
    for (size_t i = 0; i < seqlen; ++i) {
        int64_t pi = pos_ids[i];
        for (size_t h = 0; h < head; ++h) {
            const size_t offset_a = i * stride_0 + h * stride_1;
            const size_t offset_b = offset_a + (d >> 1);
            const T *in_a = in + offset_a;
            const T *in_b = in + offset_b;
            T *out_a = out + offset_a;
            T *out_b = out + offset_b;
            for (size_t j = 0; j < d / 2; ++j) {
                const double den = std::pow(theta, 2.0 * j / d);
                const double phi = (double) pi / den;
                const float cos_phi = std::cos(phi);
                const float sin_phi = std::sin(phi);

                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out_a[j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_a[j]) * cos_phi - llaisys::utils::cast<float>(in_b[j]) * sin_phi);
                    out_b[j] = llaisys::utils::cast<T>(llaisys::utils::cast<float>(in_b[j]) * cos_phi + llaisys::utils::cast<float>(in_a[j]) * sin_phi);
                } else {
                    out_a[j] = in_a[j] * cos_phi - in_b[j] * sin_phi;
                    out_b[j] = in_b[j] * cos_phi + in_a[j] * sin_phi;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta, llaisysDataType_t dtype, const size_t &seqlen, const size_t &head, const size_t &d) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                    reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, head, d);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, head, d);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                    reinterpret_cast<const int64_t *>(pos_ids), theta, seqlen, head, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
