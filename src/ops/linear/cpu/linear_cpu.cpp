#include "linear_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

// 有bias的
template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, const size_t &n, const size_t &m, const size_t &t) {
    for (size_t i = 0, p = 0, h = 0; i < n; ++i, h += t) {
        for (size_t j = 0, q = 0; j < m; ++j, ++p) {
            float acc = llaisys::utils::cast<float>(bias[j]);
            for (size_t k = 0, hh = h; k < t; ++k, ++q, ++hh) {
                const float a = llaisys::utils::cast<float>(in[hh]);
                const float b = llaisys::utils::cast<float>(weight[q]);
                acc += a * b;
            }
            out[p] = llaisys::utils::cast<T>(acc);
        }
    }
}

// 没bias的
template <typename T>
void linear_(T *out, const T *in, const T *weight, const size_t &n, const size_t &m, const size_t &t) {
    for (size_t i = 0, p = 0, h = 0; i < n; ++i, h += t) {
        for (size_t j = 0, q = 0; j < m; ++j, ++p) {
            float acc = 0.f;
            for (size_t k = 0, hh = h; k < t; ++k, ++q, ++hh) {
                const float a = llaisys::utils::cast<float>(in[hh]);
                const float b = llaisys::utils::cast<float>(weight[q]);
                acc += a * b;
            }
            out[p] = llaisys::utils::cast<T>(acc);
        }
    }
}

namespace llaisys::ops::cpu {
// 有bias的
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t dtype,
            const size_t &n, const size_t &m, const size_t &t){
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), 
                        reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), n, m, t);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), n, m, t);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), 
                        reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), n, m, t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
// 没有bias的
void linear(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t dtype, const size_t &n, const size_t &m, const size_t &t) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in), reinterpret_cast<const float *>(weight), n, m, t);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), reinterpret_cast<const llaisys::bf16_t *>(weight), n, m, t);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), reinterpret_cast<const llaisys::fp16_t *>(weight), n, m, t);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::cpu
