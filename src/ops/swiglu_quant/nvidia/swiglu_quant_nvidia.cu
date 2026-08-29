#include "swiglu_quant_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_quant.cuh"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace {

// 每 block 处理的最大元素数（128 的倍数）。decode(m=1) 时大 k 行拆成多 block 提升并行度；
// 小 k 行单段（smem ≤ 8KB），行为与旧版一致。
constexpr size_t kSeg = 2048;
constexpr size_t kSegMaxBlocks = 16;  // 每行最多 16 段，避免极端 grid

// 公共：向量化 swiglu（load_4d 读 gate/up）+ Hadamard 段内旋转，写 s_seg，返回段内 absmax
template <typename T>
__device__ __forceinline__ float swiglu_hadamard_segment(float *s_seg, const T *g, const T *u,
                                                         size_t seg_len, size_t tid, size_t block_dim) {
    // 1. swiglu：每线程 float4（4 元素）向量化读 gate/up
    const size_t nvec = seg_len / 4;
    for (size_t i = tid; i < nvec; i += block_dim) {
        const float4 gv = llaisys::utils::nvidia::load_4d(g + i * 4);
        const float4 uv = llaisys::utils::nvidia::load_4d(u + i * 4);
        float4 s;
        s.x = uv.x * gv.x / (1.0f + expf(-gv.x));
        s.y = uv.y * gv.y / (1.0f + expf(-gv.y));
        s.z = uv.z * gv.z / (1.0f + expf(-gv.z));
        s.w = uv.w * gv.w / (1.0f + expf(-gv.w));
        llaisys::utils::nvidia::save_4d(s_seg + i * 4, s);
    }
    __syncthreads();

    // 2. Hadamard：段内每 128 元素块一个 warp 蝶形（128 块间独立，可安全跨段）
    const size_t warp = tid >> 5, lane = tid & 31, nwarps = block_dim >> 5;
    const size_t nblocks = seg_len / 128;
    for (size_t b = warp; b < nblocks; b += nwarps) {
        float reg[4];
        float *blk = s_seg + b * 128;
#pragma unroll
        for (int c = 0; c < 4; ++c) reg[c] = blk[lane * 4 + c];
        llaisys::utils::nvidia::hadamard_128_warp(reg, lane);
#pragma unroll
        for (int c = 0; c < 4; ++c) blk[lane * 4 + c] = reg[c];
    }
    __syncthreads();

    // 3. 段 absmax
    float local = 0.f;
    for (size_t j = tid; j < seg_len; j += block_dim) {
        local = fmaxf(local, fabsf(s_seg[j]));
    }
    return local;
}

// 块内归并段 absmax
__device__ __forceinline__ float block_max_reduce(float local, size_t tid, size_t block_dim) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, off));
    }
    __shared__ float warp_max[16];
    const size_t warp = tid >> 5;
    const size_t nwarps = block_dim >> 5;
    if ((tid & 31) == 0) warp_max[warp] = local;
    __syncthreads();
    float m = 0.f;
    if (tid < 32) {
        m = (tid < nwarps) ? warp_max[tid] : 0.f;  // 只读有效 warp，避免未初始化 smem
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, off));
        }
    }
    return m;
}

// scan：swiglu + Hadamard + 段 absmax → 原子归并整行（absmax≥0，float 位模式 int 序 = 值序）
template <typename T>
__global__ void swiglu_quant_scan_kernel(float *row_max, const T *gate, const T *up, size_t k) {
    extern __shared__ float s_seg[];
    const size_t row = blockIdx.y;
    const size_t seg_begin = blockIdx.x * kSeg;
    const size_t seg_len = min(k, seg_begin + kSeg) - seg_begin;
    const T *g = gate + row * k + seg_begin;
    const T *u = up + row * k + seg_begin;
    const float local = swiglu_hadamard_segment(s_seg, g, u, seg_len, threadIdx.x, blockDim.x);
    const float m = block_max_reduce(local, threadIdx.x, blockDim.x);
    if (threadIdx.x == 0) {
        atomicMax(reinterpret_cast<int *>(&row_max[row]), __float_as_int(m));
    }
}

// emit：swiglu + Hadamard + 用整行 scale 量化；blockIdx.x==0 的 block 写 out_scale
template <typename T>
__global__ void swiglu_quant_emit_kernel(int8_t *out_q, float *out_scale, const float *row_max,
                                         const T *gate, const T *up, size_t k) {
    extern __shared__ float s_seg[];
    const size_t row = blockIdx.y;
    const size_t seg_begin = blockIdx.x * kSeg;
    const size_t seg_len = min(k, seg_begin + kSeg) - seg_begin;
    const T *g = gate + row * k + seg_begin;
    const T *u = up + row * k + seg_begin;
    swiglu_hadamard_segment(s_seg, g, u, seg_len, threadIdx.x, blockDim.x);

    const float amax = row_max[row];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        out_scale[row] = (amax == 0.f) ? 1.f : (amax / 127.f);
    }
    const float inv = 1.f / ((amax == 0.f) ? 1.f : (amax / 127.f));
    int8_t *row_q = out_q + row * k + seg_begin;
    // 量化：每线程 4 个 int8 打包写（4B 对齐，k 为 128 倍数）
    const size_t nvec = seg_len / 4;
    for (size_t i = threadIdx.x; i < nvec; i += blockDim.x) {
        int q[4];
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            const float v = s_seg[i * 4 + c] * inv;
            q[c] = static_cast<int>(nearbyintf(v));
            q[c] = max(-128, min(127, q[c]));
        }
        const int packed = (q[0] & 0xFF) | ((q[1] & 0xFF) << 8) |
                           ((q[2] & 0xFF) << 16) | ((q[3] & 0xFF) << 24);
        reinterpret_cast<int *>(row_q)[i] = packed;
    }
}

template <typename T>
void launch(std::byte *out_q, std::byte *out_scale, const std::byte *gate, const std::byte *up, size_t m, size_t k) {
    const int threads = 256;
    const size_t nseg = min((k + kSeg - 1) / kSeg, kSegMaxBlocks);

    // 整行 absmax 归并缓冲（跨 scan/emit 两 kernel）
    static float *d_row_max = nullptr;
    static size_t d_row_max_cap = 0;
    if (d_row_max == nullptr || d_row_max_cap < m) {
        if (d_row_max) cudaFree(d_row_max);
        CUDA_CHECK(cudaMalloc(&d_row_max, m * sizeof(float)));
        d_row_max_cap = m;
    }
    CUDA_CHECK(cudaMemset(d_row_max, 0, m * sizeof(float)));  // 同步清零，避免 stream 竞争

    dim3 grid(static_cast<unsigned>(nseg), static_cast<unsigned>(m));
    const size_t smem = kSeg * sizeof(float);  // 每 block 最大段 2048 元素 → 8KB

    swiglu_quant_scan_kernel<T><<<grid, threads, smem>>>(
        d_row_max, reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), k);
    CUDA_CHECK(cudaGetLastError());

    swiglu_quant_emit_kernel<T><<<grid, threads, smem>>>(
        reinterpret_cast<int8_t *>(out_q), reinterpret_cast<float *>(out_scale), d_row_max,
        reinterpret_cast<const T *>(gate), reinterpret_cast<const T *>(up), k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

namespace llaisys::ops::nvidia {

void swiglu_quant(std::byte *out_q, std::byte *out_scale, const std::byte *gate, const std::byte *up,
                  llaisysDataType_t in_dtype, size_t m, size_t k) {
    switch (in_dtype) {
    case LLAISYS_DTYPE_F32:
        return launch<float>(out_q, out_scale, gate, up, m, k);
    case LLAISYS_DTYPE_F16:
        return launch<llaisys::fp16_t>(out_q, out_scale, gate, up, m, k);
    case LLAISYS_DTYPE_BF16:
        return launch<llaisys::bf16_t>(out_q, out_scale, gate, up, m, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
    }
}

} // namespace llaisys::ops::nvidia
