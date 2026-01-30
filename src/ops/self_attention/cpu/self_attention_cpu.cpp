#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

template <typename T>
void self_attention_(T *attn_val, const T *Q, const T *K, const T *V, const float &scale,
                     const size_t &seqlen, const size_t &nhead, const size_t &dv, const size_t &d, const size_t &total_len, const size_t &nkvhead) {
    if (total_len == 0 || seqlen == 0 || nhead == 0 || nkvhead == 0) {
        return;
    }
    std::vector<float> tmp(total_len);
    std::vector<float> elems(total_len);
    /*
        这里得参考的pytorch中的实现 一组连续的q/k要对应连续的映射
        原来取模的实现会导致交替映射 与pytorch中的实现不符
    */
    const size_t kv_repeat = (nkvhead != 0 && nhead % nkvhead == 0) ? (nhead / nkvhead) : 0;

    for (size_t nh = 0; nh < nhead; ++ nh) {
        const size_t nkvh = (kv_repeat > 0) ? (nh / kv_repeat) : (nh % nkvhead);
        for (size_t i = 0; i < seqlen; ++ i) {
            const size_t limit = std::min(i + std::max((size_t)0, total_len - seqlen), total_len - 1);
            const size_t base_q = (i * nhead + nh) * d;
            const size_t base_a = (i * nhead + nh) * dv;

            // Q(seqlen, nhead, d) x K^T(total_len, nkvhead, d) => T(seqlen, nhead, total_len)
            float max_elem = -INFINITY;
            for (size_t j = 0; j < total_len; ++j) {
                const size_t base_k = (j * nkvhead + nkvh) * d;
                float elem = 0.0f;
                for (size_t k = 0; k < d; ++k) {
                    const size_t offset_q = base_q + k;
                    const size_t offset_k = base_k + k;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        elem += llaisys::utils::cast<float>(Q[offset_q]) * llaisys::utils::cast<float>(K[offset_k]);
                    } else {
                        elem += Q[offset_q] * K[offset_k];
                    }
                }
                elem *= scale;
                elems[j] = elem;
                if (j <= limit) {
                    max_elem = std::max(max_elem, elem);
                }
            }

            // causal softmax
            float sum = 0.0f;
            for (size_t j = 0; j <= limit; ++j) {
                const float elem = std::exp(elems[j] - max_elem);
                elems[j] = elem;
                sum += elem;
            }
            for (size_t j = 0; j < total_len; ++j) {
                tmp[j] = (j <= limit && sum > 0.0f) ? elems[j] / sum : 0.0f;
            }

            // T(seqlen, nhead, total_len) X V(total_len, nkvhead, dv) => A(seqlen, nhead, dv)
            for (size_t j = 0; j < dv; ++ j) {
                float elem = 0.0f;
                for (size_t k = 0; k < total_len; ++ k) {
                    const size_t offset_v = (k * nkvhead + nkvh) * dv + j;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        elem += tmp[k] * llaisys::utils::cast<float>(V[offset_v]);
                    } else {
                        elem += tmp[k] * V[offset_v];
                    }
                }
                const size_t offset_a = base_a + j;
                attn_val[offset_a] = llaisys::utils::cast<T>(elem);
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k, const std::byte *v, const float &scale, llaisysDataType_t dtype,
                    const size_t &seqlen, const size_t &nhead, const size_t &dv, const size_t &d, const size_t &total_len, const size_t &nkvhead) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val), reinterpret_cast<const float *>(q), reinterpret_cast<const float *>(k),
                               reinterpret_cast<const float *>(v), scale, seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val), reinterpret_cast<const llaisys::bf16_t *>(q), reinterpret_cast<const llaisys::bf16_t *>(k),
                               reinterpret_cast<const llaisys::bf16_t *>(v), scale, seqlen, nhead, dv, d, total_len, nkvhead);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val), reinterpret_cast<const llaisys::fp16_t *>(q), reinterpret_cast<const llaisys::fp16_t *>(k),
                               reinterpret_cast<const llaisys::fp16_t *>(v), scale, seqlen, nhead, dv, d, total_len, nkvhead);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
