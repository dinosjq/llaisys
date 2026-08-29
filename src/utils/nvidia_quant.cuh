#pragma once

#include "nvidia_utils.cuh"

#include <cstddef>
#include <cstdint>

namespace llaisys::utils::nvidia {

// 公共量化设备函数 (quantize_act / rms_norm_quant / swiglu_quant)
// 约定: 1 block = 1 行, blockDim 个线程: s_row 已含 producer 结果(旋转前), k 必须为 128 的倍数。

constexpr size_t kHBlock = 128;                   // Hadamard 块大小 (K 维按 128 分组)
constexpr float kInvSqrtH = 0.08838834764831845f; // 1/sqrt(128)

// warp 内 128 元素 Walsh-Hadamard 蝶形 (每 lane 4 元素 reg[4], 元素 index = lane * 4 + c)
// 蝶形用 __shfl_xor 在寄存器内交换，无每轮 __syncthreads，块间仅 1 次同步
__device__ __forceinline__ void hadamard_128_warp(float reg[4], int lane) {
    // 自底向上每次处理一个 2^b 的子块
    for (int bit = 1; bit < (int)kHBlock; bit <<= 1) {
        if (bit < 4) {
            // 同 lane: 元素 c 与 c^bit 在寄存器内交换
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if ((c & bit) == 0) {
                    const float a = reg[c];
                    const float b = reg[c | bit];
                    reg[c] = a + b;
                    reg[c | bit] = a - b;
                }
            }
        } else {
            // 跨 lane: 元素 index = lane * 4 + c 的 bit 位对应 lane 的 (bit/4) 位, 用 __shfl_xor 交换
            const int off = bit >> 2;
#pragma unroll
            for (int c = 0; c < 4; ++c) {
                if ((lane & off) != 0) { // idx 的 bit 位 = 1 (b 侧)
                    const float a = __shfl_xor_sync(0xffffffffu, reg[c], off);
                    reg[c] = a - reg[c];
                } else { // a 侧
                    const float b = __shfl_xor_sync(0xffffffffu, reg[c], off);
                    reg[c] = reg[c] + b;
                }
            }
        }
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        reg[c] *= kInvSqrtH;
    }
}

// 对 smem 行 s_row 做 Hadamard 旋转 + per-row absmax => scale => round, 写 out_q / out_scale。
// s_row 在调用前必须已被 producer 填好 (并已 __syncthreads 保证可见)
__device__ __forceinline__ void quantize_smem_row(float *s_row, 
                                                  int8_t *out_q, 
                                                  float *out_scale, 
                                                  size_t k,
                                                  size_t row, 
                                                  size_t tid, 
                                                  size_t block_dim) 
{
    const size_t warp = tid >> 5;
    const size_t nwarps = block_dim >> 5;
    const int lane = static_cast<int>(tid & 31);

    // 1. 每 128 块由一个 warp 做 shuffle 蝶形，nwarps 个 warp 并行；块间仅 1 次同步
    const size_t nblocks = k / kHBlock;
    for (size_t b = warp; b < nblocks; b += nwarps) {
        float reg[4];
        float *blk = s_row + b * kHBlock;
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            reg[c] = blk[lane * 4 + c];
        }
        hadamard_128_warp(reg, lane);
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            blk[lane * 4 + c] = reg[c];
        }
    }
    __syncthreads();

    // 2. 每行 absmax → scale
    float local = 0.f;
    for (size_t j = tid; j < k; j += block_dim) {
        local = fmaxf(local, fabsf(s_row[j]));
    }
    __shared__ float warp_max[32];
    __shared__ float row_scale;
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    }
    if (lane == 0) {
        warp_max[warp] = local;
    }
    __syncthreads();
    float amax = 0.f;
    if (tid < (block_dim + 31) / 32) {
        amax = warp_max[tid];
    }
    if (tid < 32) {
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
        }
    }
    if (tid == 0) {
        row_scale = (amax == 0.f) ? 1.f : (amax / 127.f);
        out_scale[row] = row_scale;
    }
    __syncthreads();

    // 3. 量化
    int8_t *row_q = out_q + row * k;
    const float inv = 1.f / row_scale;
    for (size_t j = tid; j < k; j += block_dim) {
        const float q = nearbyintf(s_row[j] * inv);
        row_q[j] = static_cast<int8_t>(max(-128, min(127, static_cast<int>(q))));
    }
}

} // namespace llaisys::utils::nvidia
