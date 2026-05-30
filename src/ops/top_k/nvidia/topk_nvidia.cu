#include "../../../utils.hpp"
#include "topk_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <cmath>

constexpr size_t RADIX_BITS = 2;
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
#pragma unroll
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

template <typename T>
void _topk_launch(std::byte *out_idx, std::byte *out_val, const std::byte *vals, size_t batch_size, size_t numel, size_t k){
    size_t *d_out_idx = reinterpret_cast<size_t *>(out_idx);
    T *d_out_val = reinterpret_cast<T *>(out_val);
    const T *d_vals = reinterpret_cast<const T *>(vals);
    
    dim3 blockDim(256);
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
} // namespace llaisys::ops::cpu
