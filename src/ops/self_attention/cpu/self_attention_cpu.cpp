#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <algorithm>
#include <cmath>

template <typename T>
void self_attention_(T *attn_val, const T *Q, const T *K, const T *V, const float &scale,
                     const size_t &seqlen, const size_t &nhead, const size_t &dv, const size_t &d, const size_t &total_len, const size_t &nkvhead) {
    if (total_len == 0 || seqlen == 0 || nhead == 0 || nkvhead == 0) {
        return;
    }
    float *tmp = (float *)malloc(sizeof(float) * seqlen * nhead * total_len);
    float *elems = (float *)malloc(sizeof(float) * total_len);

    // Q(seqlen, nhead, d) x K^T(total_len, nkvhead, d) => T(seqlen, nhead, total_len)
    /*
        这里得参考的pytorch中的实现 一组连续的q/k要对应连续的映射
        原来取模的实现会导致交替映射 与pytorch中的实现不符
    */
    const size_t kv_repeat = (nkvhead != 0 && nhead % nkvhead == 0) ? (nhead / nkvhead) : 0;
    for (size_t h = 0; h < nhead; ++h) {
        const size_t kvh = (kv_repeat > 0) ? (h / kv_repeat) : (h % nkvhead);
        for (size_t i = 0; i < seqlen; ++i) {
            const size_t limit = std::min(i + std::max((size_t)0, total_len - seqlen), total_len - 1);
            float max_elem = -INFINITY;
            // matrix multipy Q x K
            for (size_t j = 0; j < total_len; ++j) {
                float elem = 0.0f;
                for (size_t k = 0; k < d; ++k) {
                    const size_t offset_q = i * nhead * d + h * d + k;
                    const size_t offset_k = j * nkvhead * d + kvh * d + k;
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
            // causalsoftmax
            float sum = 0.0f;
            for (size_t j = 0; j <= limit; ++j) {
                const float t = std::exp(elems[j] - max_elem);
                elems[j] = t;
                sum += t;
            }
            for (size_t j = 0; j < total_len; ++j) {
                const size_t offset_t = i * nhead * total_len + h * total_len + j;
                if (j <= limit && sum > 0.0f) {
                    tmp[offset_t] = elems[j] / sum;
                } else {
                    tmp[offset_t] = 0.0f;
                }
            }
        }
    }

    // T(seqlen, nhead, total_len) X V(total_len, nkvhead, dv) => A(seqlen, nhead, dv)
    for (size_t h = 0; h < nhead; ++h) {
        const size_t kvh = (kv_repeat > 0) ? (h / kv_repeat) : (h % nkvhead);
        for (size_t i = 0; i < seqlen; ++i) {
            for (size_t j = 0; j < dv; ++j) {
                float elem = 0.0f;
                for (size_t k = 0; k < total_len; ++k) {
                    const size_t offset_t = i * nhead * total_len + h * total_len + k;
                    const size_t offset_v = k * nkvhead * dv + kvh * dv + j;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        elem += tmp[offset_t] * llaisys::utils::cast<float>(V[offset_v]);
                    } else {
                        elem += tmp[offset_t] * V[offset_v];
                    }
                }
                const size_t offset_a = i * nhead * dv + h * dv + j;
                attn_val[offset_a] = llaisys::utils::cast<T>(elem);
            }
        }
    }

    free(elems);
    free(tmp);
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
