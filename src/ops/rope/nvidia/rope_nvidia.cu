#include "rope_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

template <typename T>
__global__ void rope_kernel(T *__restrict__ _out, const T *__restrict__ _in, const int64_t *__restrict__ _pos_ids,
                            const float _theta, const size_t _seqlen, const size_t _head, const size_t _d) {
    const int i = blockIdx.y;
    const int h = blockIdx.x;
    const int tid = threadIdx.x;

    const int hf_d = _d >> 1;
    const int offset = ((i * _head) + h) * _d;

    const float inv_d = -1.0f / static_cast<float>(_d);
    const float pi = static_cast<float>(_pos_ids[i]);

    const T *in_a = _in + offset;
    const T *in_b = in_a + hf_d;
    T *out_a = _out + offset;
    T *out_b = out_a + hf_d;

    for (int col = (tid << 2); col < hf_d; col += 128) {
        if (col + 3 < hf_d) {
            const float den_0 = powf(_theta, static_cast<float>((col | 0) << 1) * inv_d);
            const float phi_0 = pi * den_0;
            const float cos_phi_0 = cosf(phi_0);
            const float sin_phi_0 = sinf(phi_0);

            const float den_1 = powf(_theta, static_cast<float>((col | 1) << 1) * inv_d);
            const float phi_1 = pi * den_1;
            const float cos_phi_1 = cosf(phi_1);
            const float sin_phi_1 = sinf(phi_1);

            const float den_2 = powf(_theta, static_cast<float>((col | 2) << 1) * inv_d);
            const float phi_2 = pi * den_2;
            const float cos_phi_2 = cosf(phi_2);
            const float sin_phi_2 = sinf(phi_2);

            const float den_3 = powf(_theta, static_cast<float>((col | 3) << 1) * inv_d);
            const float phi_3 = pi * den_3;
            const float cos_phi_3 = cosf(phi_3);
            const float sin_phi_3 = sinf(phi_3);

            const float4 flo4_a = llaisys::utils::nvidia::load_4d(in_a + col);
            const float4 flo4_b = llaisys::utils::nvidia::load_4d(in_b + col);

            const float4 flo4_va = make_float4(flo4_a.x * cos_phi_0 - flo4_b.x * sin_phi_0,
                                               flo4_a.y * cos_phi_1 - flo4_b.y * sin_phi_1,
                                               flo4_a.z * cos_phi_2 - flo4_b.z * sin_phi_2,
                                               flo4_a.w * cos_phi_3 - flo4_b.w * sin_phi_3);
            const float4 flo4_vb = make_float4(flo4_b.x * cos_phi_0 + flo4_a.x * sin_phi_0,
                                               flo4_b.y * cos_phi_1 + flo4_a.y * sin_phi_1,
                                               flo4_b.z * cos_phi_2 + flo4_a.z * sin_phi_2,
                                               flo4_b.w * cos_phi_3 + flo4_a.w * sin_phi_3);

            llaisys::utils::nvidia::save_4d(out_a + col, flo4_va);
            llaisys::utils::nvidia::save_4d(out_b + col, flo4_vb);
        } else {
            for (int j = col; j < hf_d; ++j) {
                const float den = std::pow(_theta, static_cast<float>(j << 1) * inv_d);
                const float phi = pi / den;
                const float cos_phi = cosf(phi);
                const float sin_phi = sin(phi);

                const float val_a = llaisys::utils::nvidia::cast<float>(in_a[j]);
                const float val_b = llaisys::utils::nvidia::cast<float>(in_b[j]);

                const float res_a = val_a * cos_phi - val_b * sin_phi;
                const float res_b = val_b * cos_phi + val_a * sin_phi;

                out_a[j] = llaisys::utils::nvidia::cast<T>(res_a);
                out_b[j] = llaisys::utils::nvidia::cast<T>(res_b);
            }
        }
    }
}

template <typename T>
void rope_launch(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta,
                 const size_t &seqlen, const size_t &head, const size_t &d) {
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_in = reinterpret_cast<const T *>(in);
    const auto *d_pos_ids = reinterpret_cast<const int64_t *>(pos_ids);

    dim3 blockDim(32);
    dim3 gridDim(head, seqlen);

    rope_kernel<<<gridDim, blockDim>>>(d_out, d_in, d_pos_ids, theta, seqlen, head, d);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, const float &theta, llaisysDataType_t type,
          const size_t &seqlen, const size_t &head, const size_t &d) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_launch<float>(out, in, pos_ids, theta, seqlen, head, d);
    case LLAISYS_DTYPE_BF16:
        return rope_launch<llaisys::bf16_t>(out, in, pos_ids, theta, seqlen, head, d);
    case LLAISYS_DTYPE_F16:
        return rope_launch<llaisys::fp16_t>(out, in, pos_ids, theta, seqlen, head, d);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
