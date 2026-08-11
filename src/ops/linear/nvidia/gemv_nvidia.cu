#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#include "utils/nvidia_utils.cuh"
#include "utils.hpp"

__device__ __forceinline__ float warp_reduce_gemv(float val) {
#pragma unroll
    for (size_t offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

static constexpr size_t GEMV_BLOCK_DIM = 256;   // 8 warps
static constexpr size_t GEMV_NWARPS = 8;        // 每 warp 1 个 out 行
static constexpr size_t GEMV_M_MAX = 64;        // decode batch 上限

// ── GEMV: out[m, n] = in[m, k] @ weight[n, k]^T (+ bias[n]) ────
// 参照 v3 flash-decoding 的优化：模板化、cp.async 流式、全体 warp 参与。
//   * grid = (ceil(n / NWARPS), 1)：每 block 处理 NWARPS=8 个 out 行 × 全部 m
//   * 每 warp 负责 1 个 out 行（nb），对全部 m 做 k 维 dot
//   * weight 行用 cp.async 流式加载到 smem（NUM_STAGES 深流水，全体线程分摊）
//   * in[m, k] 从 global 直接读（m×k 小，L2 命中，各 warp 共享）
//   * acc 存 smem（s_out[warp][m]），跨 k tile 累加
// 关键：无 producer/consumer 分离——每 warp 发自己 weight 行的 cp.async、
//       等自己的 load、算自己的 out 行（同 v3 全体参与模型）。
template <typename T, size_t K_TILE, size_t NUM_STAGES>
__global__ void gemv_kernel(
    T *__restrict__ _out, const T *__restrict__ _in, const T *__restrict__ _weight,
    const T *__restrict__ _bias, const size_t _m, const size_t _n, const size_t _k)
{
    const size_t tid     = threadIdx.x;
    const size_t warp_id = tid >> 5;
    const size_t lane_id = tid & 31;
    const size_t nb      = blockIdx.x * GEMV_NWARPS + warp_id;   // 本 warp 的 out 行
    if (nb >= _n) return;

    // smem: weight tile 多 stage + out acc
    extern __shared__ __align__(16) char _smem[];
    T     *s_w = reinterpret_cast<T *>(_smem);                 // NUM_STAGES*NWARPS*K_TILE
    float *s_out = reinterpret_cast<float *>(s_w + NUM_STAGES * GEMV_NWARPS * K_TILE);  // NWARPS*M_MAX

    // acc 初始化（本 warp 的 m 个）
    float *acc = s_out + warp_id * GEMV_M_MAX;
    for (size_t mm = lane_id; mm < GEMV_M_MAX; mm += 32) acc[mm] = 0.0f;
    __syncthreads();

    const T *w_row = _weight + nb * _k;   // 本 warp 的 weight 行

    // cp.async: 全体线程分摊发 weight 行块（16B）
    constexpr size_t ELEMS_PER_BLK = 16 / sizeof(T);
    constexpr size_t BLOCKS_PER_ROW = K_TILE / ELEMS_PER_BLK;

    auto cp_load = [&](size_t stage, size_t k_start) {
        const size_t rem = (k_start + K_TILE <= _k) ? K_TILE : (_k - k_start);
        // 每 warp 用自己的 lane 分摊 load 自己的 weight 行
        for (size_t idx = lane_id; idx < BLOCKS_PER_ROW; idx += 32) {
            const size_t off = idx * ELEMS_PER_BLK;
            if (off < rem) {
                T *dst = s_w + (stage * GEMV_NWARPS + warp_id) * K_TILE + off;
                __pipeline_memcpy_async(dst, w_row + k_start + off, ELEMS_PER_BLK * sizeof(T));
            }
        }
    };

    // ── 预取前 NUM_STAGES 个 k tile ────────────────────────────
    const size_t num_tiles = (_k + K_TILE - 1) / K_TILE;
    const size_t nprefetch = (num_tiles < NUM_STAGES) ? num_tiles : NUM_STAGES;
    for (size_t s = 0; s < nprefetch; ++s) {
        cp_load(s, s * K_TILE);
        __pipeline_commit();
    }

    // ── 流水主循环（lookahead NUM_STAGES-1，prefetch stage i+NUM_STAGES-1）──
    for (size_t i = 0; i < num_tiles; ++i) {
        if (i + (NUM_STAGES - 1) < num_tiles) {
            cp_load((i + NUM_STAGES - 1) % NUM_STAGES, (i + NUM_STAGES - 1) * K_TILE);
        }
        __pipeline_commit();

        const size_t future = num_tiles - i - 1;
        __pipeline_wait_prior((future < NUM_STAGES - 1) ? future : NUM_STAGES - 1);
        __syncthreads();

        // 计算本 warp 的 out 行：对全部 m 做 k-tile dot
        // weight 常驻寄存器（每 lane 一次读 PER_LANE 元素，16 个 mm 复用）
        // in 从 global 读（m×k 小，L2 命中；进 smem 实测变差——cp.async 开销 > 收益）
        const size_t k_start = i * K_TILE;
        const size_t k_end = (k_start + K_TILE <= _k) ? k_start + K_TILE : _k;
        constexpr size_t PER_LANE = K_TILE / 32;
        const T *w_tile = s_w + (i % NUM_STAGES) * GEMV_NWARPS * K_TILE + warp_id * K_TILE;
        float w_reg[PER_LANE];
#pragma unroll
        for (size_t e = 0; e < PER_LANE; ++e)
            w_reg[e] = llaisys::utils::nvidia::cast<float>(w_tile[lane_id * PER_LANE + e]);
        for (size_t mm = 0; mm < _m; ++mm) {
            float dot = 0.0f;
            const T *in_row = _in + mm * _k + k_start;
            if (k_start + K_TILE <= k_end) {   // 完整 tile
#pragma unroll
                for (size_t e = 0; e < PER_LANE; ++e)
                    dot += llaisys::utils::nvidia::cast<float>(in_row[lane_id * PER_LANE + e]) * w_reg[e];
            } else {                           // 不完整 tile
#pragma unroll
                for (size_t e = 0; e < PER_LANE; ++e)
                    if (k_start + lane_id * PER_LANE + e < k_end)
                        dot += llaisys::utils::nvidia::cast<float>(in_row[lane_id * PER_LANE + e]) * w_reg[e];
            }
            dot = warp_reduce_gemv(dot);
            acc[mm] += dot;
        }
        __syncthreads();   // stage 读完，下一轮才能覆盖
    }

    // ── 写输出 out[mm, nb] = acc[mm] + bias[nb] ────────────────
    const float bias_v = (_bias != nullptr) ? llaisys::utils::nvidia::cast<float>(_bias[nb]) : 0.0f;
    for (size_t mm = 0; mm < _m; ++mm) {
        if (lane_id == 0)
            _out[mm * _n + nb] = llaisys::utils::nvidia::cast<T>(acc[mm] + bias_v);
    }
}

// ── Host launch ────────────────────────────────────────────────
template <typename T>
void gemv_launch(T *out, const T *in, const T *weight, const T *bias,
                 const size_t m, const size_t n, const size_t k)
{
    constexpr size_t K_TILE = 256;
    constexpr size_t NUM_STAGES = 3;

    const size_t grid = (n + GEMV_NWARPS - 1) / GEMV_NWARPS;
    dim3 blockDim(GEMV_BLOCK_DIM);
    const size_t smem = NUM_STAGES * GEMV_NWARPS * K_TILE * sizeof(T)
                      + GEMV_NWARPS * GEMV_M_MAX * sizeof(float);

    auto *kernel = gemv_kernel<T, K_TILE, NUM_STAGES>;
    // smem 随 m 变化，每次 launch 都需设置（不能用 static 只设一次）
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    kernel<<<grid, blockDim, smem>>>(out, in, weight, bias, m, n, k);
    CUDA_CHECK(cudaGetLastError());
}

// ── C wrapper（实验/对比用）────────────────────────────────────
extern "C" __export void llaisysGemv(
    void *out, void *in, void *weight, void *bias,
    size_t m, size_t n, size_t k, int dtype)
{
    switch (dtype) {
    case 0: gemv_launch<float>(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
        reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, n, k); break;
    case 1: gemv_launch<llaisys::fp16_t>(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
        reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k); break;
    case 2: gemv_launch<llaisys::bf16_t>(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
        reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k); break;
    default: ASSERT(false, "gemv: bad dtype");
    }
    CUDA_CHECK(cudaGetLastError());
}
