#include "../../../utils.hpp"
#include "topk_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cmath>

constexpr size_t RADIX_BITS = 4;
constexpr size_t RADIX_SIZE = 1 << RADIX_BITS;

// float <---> uint32
__device__ __forceinline__ unsigned int _float_2_uint(float v){
    unsigned int x = __float_as_uint(v);
    unsigned int mask = (x & 0x80000000) ? 0xffffffff : 0x80000000;
    unsigned int res  = x ^ mask;
    return res;
}

__device__ __forceinline__ float _uint_2_float(unsigned int v){
    unsigned int mask = (v & 0x80000000) ? 0x80000000 : 0xffffffff;
    unsigned int res = v ^ mask;
    return __uint_as_float(res);
}

__device__ __forceinline__ unsigned int _extract(unsigned int v, unsigned int pos, unsigned int len){
    unsigned int mask = (1U << len) - 1;
    return (v >> pos) & mask;
}

template <typename T>
__global__ void _topk_kernel(size_t *out_idx, T *out_val, const T *vals, const size_t numel, const size_t k){
    // find k_th
    __shared__ unsigned int block_counts[RADIX_SIZE];
    __shared__ unsigned int index[1];
    unsigned int warp_counts[RADIX_SIZE];
    const size_t stride = blockDim.x;
    const size_t tid = threadIdx.x;
    const size_t lane_id = tid & 31;
    const size_t warp_id = tid >> 5;
    const size_t batch_id = blockIdx.x;
    const size_t in_offset = batch_id * numel;
    const size_t out_offset = batch_id * k;
    unsigned int kth_val = 0;
    unsigned int mask = 0;
    size_t rank = k;
    if(tid == 0){
        index[0] = 0;
    }
#pragma unroll
    for(int pos = sizeof(float) * 8 - RADIX_BITS; pos >= 0; pos -= RADIX_BITS){
#pragma unroll
        for(int i = 0; i < RADIX_SIZE; ++ i){
            warp_counts[i] = 0;
        }
        if(lane_id == 0 && warp_id < RADIX_SIZE){
            block_counts[warp_id] = 0;
        }
        __syncthreads();
        /** 非归约写法 */
//         for(size_t i = tid; i - lane_id < numel; i += stride){
//             float fval = i < numel ? llaisys::utils::nvidia::cast<float>(vals[in_offset + i]) : 0;
//             unsigned int val = _float_2_uint(fval);
//             bool prefix = (val & mask) == kth_val;
//             unsigned int chunk = _extract(val, pos, RADIX_BITS);
//             // __ballot_sync(mask, predict) 返回一个表示 warp 中满足要求的线程状态表示的二进制串
//             unsigned int active = __ballot_sync(0xffffffff, prefix);
// #pragma unroll
//             for(size_t v = 0; v < RADIX_SIZE; ++ v){
//                 bool flag = prefix && (chunk == v);
//                 unsigned int ballot = __ballot_sync(active, flag);
//                 // __popc 返回一个二进制中 1 的个数
//                 warp_counts[v] += __popc(ballot);
//             }
//         }
        /** 归约写法 */
        for(size_t i = tid; i < numel; i += stride){
            float fval = llaisys::utils::nvidia::cast<float>(vals[in_offset + i]);
            unsigned int val = _float_2_uint(fval);
            bool prefix = (val & mask) == kth_val;
            unsigned int chunk = _extract(val, static_cast<unsigned int>(pos), static_cast<unsigned int>(RADIX_BITS));
            warp_counts[chunk] += prefix;
        }
#pragma unroll
        for(int v = 0; v < RADIX_SIZE; ++ v){
            // Warp reduce
            for(int offset = 16; offset > 0; offset >>= 1){
                warp_counts[v] += __shfl_down_sync(0xFFFFFFFF, warp_counts[v], offset);
            }
        }
        if(lane_id == 0){
#pragma unroll
            for(int v = 0; v < RADIX_SIZE; ++ v){
                atomicAdd(block_counts + v, warp_counts[v]);
            }
        }
        __syncthreads();
#pragma unroll
        for(int v = 0; v < RADIX_SIZE; ++ v){
            warp_counts[v] = block_counts[v];
        }
#pragma unroll
        for(int v = RADIX_SIZE - 1; v >= 0; -- v){
            if(block_counts[v] >= rank){
                kth_val |= v << pos;
                break;
            }
            rank -= block_counts[v];
        }
        mask = static_cast<unsigned int>(-1) ^ ((1U << pos) - 1); 
        __syncthreads();
    }
    float f_kth_val = _uint_2_float(kth_val);
    for(size_t i = tid; i < numel; i += stride){
        T val = vals[in_offset + i];
        float fval = llaisys::utils::nvidia::cast<float>(val);
        if(fval >= f_kth_val){
            size_t j = atomicAdd(index, 1);
            if(j < k){ 
                out_idx[out_offset + j] = i;
                out_val[out_offset + j] = val;
            }
        }
    }
}

// k==1 快路径: 多 block 两级归并求 max+idx
template <typename T>
__global__ void top1_partial_kernel(float *p_max, size_t *p_idx, const T *vals,
                                    size_t numel, size_t nblocks) {
    const size_t batch = blockIdx.y;
    const size_t base = batch * numel;
    const size_t tid = threadIdx.x;
    const size_t seg = (numel + nblocks - 1) / nblocks;
    const size_t begin = blockIdx.x * seg;
    const size_t end = min(numel, begin + seg);
    float lmax = -INFINITY;
    size_t lmax_i = 0;
    for (size_t i = begin + tid; i < end; i += blockDim.x) {
        const float v = llaisys::utils::nvidia::cast<float>(vals[base + i]);
        if (v > lmax) {
            lmax = v;
            lmax_i = i;
        }
    }
    // warp reduce（max 带索引）
    for (int off = 16; off > 0; off >>= 1) {
        const float o = __shfl_down_sync(0xffffffff, lmax, off);
        const size_t oi = __shfl_down_sync(0xffffffff, lmax_i, off);
        if (o > lmax) {
            lmax = o;
            lmax_i = oi;
        }
    }
    __shared__ float s_max[16];
    __shared__ size_t s_idx[16];
    const size_t warp = tid >> 5;
    if ((tid & 31) == 0) {
        s_max[warp] = lmax;
        s_idx[warp] = lmax_i;
    }
    __syncthreads();
    if (tid < 32) {
        float m = (tid < (blockDim.x >> 5)) ? s_max[tid] : -INFINITY;
        size_t mi = (tid < (blockDim.x >> 5)) ? s_idx[tid] : 0;
        for (int off = 16; off > 0; off >>= 1) {
            const float om = __shfl_down_sync(0xffffffff, m, off);
            const size_t omi = __shfl_down_sync(0xffffffff, mi, off);
            if (om > m) {
                m = om;
                mi = omi;
            }
        }
        if (tid == 0) {
            p_max[batch * nblocks + blockIdx.x] = m;
            p_idx[batch * nblocks + blockIdx.x] = mi;
        }
    }
}

template <typename T>
__global__ void top1_final_kernel(size_t *out_idx, T *out_val, const float *p_max,
                                  const size_t *p_idx, size_t nblocks) {
    const size_t batch = blockIdx.x;
    float m = -INFINITY;
    size_t mi = 0;
    for (size_t b = 0; b < nblocks; ++b) {
        const float v = p_max[batch * nblocks + b];
        if (v > m) {
            m = v;
            mi = p_idx[batch * nblocks + b];
        }
    }
    out_val[batch] = llaisys::utils::nvidia::cast<T>(m);
    out_idx[batch] = mi;
}

template <typename T>
void _topk_launch(std::byte *out_idx, std::byte *out_val, const std::byte *vals, size_t batch_size, size_t numel, size_t k){
    size_t *d_out_idx = reinterpret_cast<size_t *>(out_idx);
    T *d_out_val = reinterpret_cast<T *>(out_val);
    const T *d_vals = reinterpret_cast<const T *>(vals);

    if (k == 1) {
        // 快路径: 多 block 两级归并求 max+idx（B300: 单 block radix 8 轮过慢）
        constexpr size_t kTop1Blocks = 32;
        static float *d_pmax = nullptr;
        static size_t *d_pidx = nullptr;
        static size_t d_cap = 0;
        if (d_pmax == nullptr || d_cap < batch_size * kTop1Blocks) {
            if (d_pmax) cudaFree(d_pmax);
            if (d_pidx) cudaFree(d_pidx);
            CUDA_CHECK(cudaMalloc(&d_pmax, batch_size * kTop1Blocks * sizeof(float)));
            CUDA_CHECK(cudaMalloc(&d_pidx, batch_size * kTop1Blocks * sizeof(size_t)));
            d_cap = batch_size * kTop1Blocks;
        }
        dim3 pgrid(kTop1Blocks, static_cast<unsigned>(batch_size));
        top1_partial_kernel<T><<<pgrid, 256>>>(d_pmax, d_pidx, d_vals, numel, kTop1Blocks);
        CUDA_CHECK(cudaGetLastError());
        top1_final_kernel<T><<<static_cast<unsigned>(batch_size), 32>>>(
            d_out_idx, d_out_val, d_pmax, d_pidx, kTop1Blocks);
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    dim3 blockDim(1024);
    dim3 gridDim(batch_size);

    _topk_kernel<<<gridDim, blockDim>>>(d_out_idx, d_out_val, d_vals, numel, k);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void topk(std::byte *out_idx, std::byte *out_val, const std::byte *vals, llaisysDataType_t type, size_t batch_size, size_t numel, size_t k){
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return _topk_launch<float>(out_idx, out_val, vals, batch_size, numel, k);
    case LLAISYS_DTYPE_BF16:
        return _topk_launch<llaisys::bf16_t>(out_idx, out_val, vals, batch_size, numel, k);
    case LLAISYS_DTYPE_F16:
        return _topk_launch<llaisys::fp16_t>(out_idx, out_val, vals, batch_size, numel, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
