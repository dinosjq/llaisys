#include "gemv_w8a16_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace {

// One-stage decode GEMV (integer domain):
//   shared: BF16/F16 act → int8 (per-row scale) → __dp4a with weight int8
//   epilogue: acc * act_scale * weight_scale + bias
// Prefill/large-M: caller uses dequant + FP linear.

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr size_t kMaxSmemBytes = 48 * 1024;

__device__ __forceinline__ float warp_sum_f(float value) {
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_amax(float value) {
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

template <typename T, int M>
__global__ void gemv_w8a16_dp4a_kernel(T *__restrict__ out, const T *__restrict__ in,
                                       const int8_t *__restrict__ weight, const float *__restrict__ w_scale,
                                       const T *__restrict__ bias, size_t n, size_t k) {
    extern __shared__ char smem[];
    int8_t *act_i8 = reinterpret_cast<int8_t *>(smem);
    float *act_scale = reinterpret_cast<float *>(act_i8 + static_cast<size_t>(M) * k);

    // ---- Phase A: quantize activations into smem (once per block) ----
#pragma unroll
    for (int row = 0; row < M; ++row) {
        const T *row_in = in + static_cast<size_t>(row) * k;
        float local = 0.f;
        for (size_t j = static_cast<size_t>(threadIdx.x); j < k; j += static_cast<size_t>(blockDim.x)) {
            local = fmaxf(local, fabsf(llaisys::utils::nvidia::cast<float>(row_in[j])));
        }
        local = warp_amax(local);

        __shared__ float warp_max[32];
        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        if (lane == 0) {
            warp_max[warp] = local;
        }
        __syncthreads();

        float amax = 0.f;
        if (static_cast<unsigned>(threadIdx.x) < ((blockDim.x + 31u) >> 5)) {
            amax = warp_max[threadIdx.x];
        }
        if (threadIdx.x < 32) {
            amax = warp_amax(amax);
        }
        __shared__ float row_scale_s;
        if (threadIdx.x == 0) {
            row_scale_s = (amax == 0.f) ? 1.f : (amax / 127.f);
            act_scale[row] = row_scale_s;
        }
        __syncthreads();

        const float inv = 1.f / row_scale_s;
        int8_t *row_q = act_i8 + static_cast<size_t>(row) * k;
        for (size_t j = static_cast<size_t>(threadIdx.x); j < k; j += static_cast<size_t>(blockDim.x)) {
            const float q = nearbyintf(llaisys::utils::nvidia::cast<float>(row_in[j]) * inv);
            row_q[j] = static_cast<int8_t>(max(-128, min(127, static_cast<int>(q))));
        }
        __syncthreads();
    }

    float a_scale[M];
#pragma unroll
    for (int row = 0; row < M; ++row) {
        a_scale[row] = act_scale[row];
    }

    // ---- Phase B: INT8×INT8 GEMV via dp4a ----
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const size_t col = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock + static_cast<size_t>(warp);
    if (col >= n) {
        return;
    }

    int acc_i[M] = {};
    const int8_t *w_row = weight + col * k;
    for (size_t k0 = static_cast<size_t>(lane) * 4; k0 < k; k0 += static_cast<size_t>(kWarpSize) * 4) {
        const int w_pack = *reinterpret_cast<const int *>(w_row + k0);
#pragma unroll
        for (int row = 0; row < M; ++row) {
            const int a_pack = *reinterpret_cast<const int *>(act_i8 + static_cast<size_t>(row) * k + k0);
            acc_i[row] = __dp4a(a_pack, w_pack, acc_i[row]);
        }
    }

    const float ws = w_scale[col];
    const float bias_v = bias == nullptr ? 0.f : llaisys::utils::nvidia::cast<float>(bias[col]);
#pragma unroll
    for (int row = 0; row < M; ++row) {
        float sum = static_cast<float>(acc_i[row]);
        sum = warp_sum_f(sum);
        if (lane == 0) {
            out[static_cast<size_t>(row) * n + col] =
                llaisys::utils::nvidia::cast<T>(sum * a_scale[row] * ws + bias_v);
        }
    }
}

template <typename T, int M>
void launch_dp4a(T *out, const T *in, const int8_t *weight, const float *scale, const T *bias, size_t n, size_t k) {
    const size_t smem = static_cast<size_t>(M) * k * sizeof(int8_t) + static_cast<size_t>(M) * sizeof(float);
    if (smem > kMaxSmemBytes) {
        throw std::runtime_error("gemv_w8a16: smem too large");
    }
    const unsigned blocks = static_cast<unsigned>((n + kWarpsPerBlock - 1) / kWarpsPerBlock);
    gemv_w8a16_dp4a_kernel<T, M><<<blocks, kThreadsPerBlock, smem>>>(out, in, weight, scale, bias, n, k);
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch(T *out, const T *in, const int8_t *weight, const float *scale, const T *bias, size_t m, size_t n,
            size_t k) {
#define LLAISYS_LAUNCH_M(value)                                                                                    \
    case value:                                                                                                    \
        return launch_dp4a<T, value>(out, in, weight, scale, bias, n, k)
    switch (m) {
        LLAISYS_LAUNCH_M(1);
        LLAISYS_LAUNCH_M(2);
        LLAISYS_LAUNCH_M(3);
        LLAISYS_LAUNCH_M(4);
    default:
        throw std::runtime_error("gemv_w8a16 requires 1 <= M <= 4");
    }
#undef LLAISYS_LAUNCH_M
}

} // namespace

namespace llaisys::ops::nvidia {

bool gemv_w8a16_supported(llaisysDataType_t dtype, size_t m, size_t, size_t k) {
    const size_t smem = static_cast<size_t>(m) * k * sizeof(int8_t) + static_cast<size_t>(m) * sizeof(float);
    return (dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16) && m > 0 && m <= 4 && k >= 4 && (k % 4) == 0
           && smem <= kMaxSmemBytes;
}

void gemv_w8a16(std::byte *out, const std::byte *in, const std::byte *weight_i8, const std::byte *scale,
                const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    const auto *w = reinterpret_cast<const int8_t *>(weight_i8);
    const auto *s = reinterpret_cast<const float *>(scale);
    if (dtype == LLAISYS_DTYPE_F16) {
        return launch(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in), w, s,
                      reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    }
    if (dtype == LLAISYS_DTYPE_BF16) {
        return launch(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in), w, s,
                      reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    }
    throw std::runtime_error("Unsupported dtype for gemv_w8a16");
}

} // namespace llaisys::ops::nvidia
