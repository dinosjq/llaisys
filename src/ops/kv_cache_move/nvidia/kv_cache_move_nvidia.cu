#include "../../../utils.hpp"
#include "kv_cache_move_nvidia.cuh"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>


template<typename T>
__global__ void kv_cache_move_kernel(T *_out,       
                                    const T *_in,       
                                    const int64_t *block_ids,
                                    const int64_t *cut_idx, 
                                    const int64_t *pos_ids,  
                                    const size_t block_size,    
                                    const size_t table_width,
                                    const size_t numel)
{
    const size_t seq_id = blockIdx.x;   // 全局 token 偏移
    const size_t tid = threadIdx.x;
    // 反查所属 batch：cut_idx[batch] <= seq_id < cut_idx[batch+1]
    size_t batch = 0;
    while (cut_idx[batch + 1] <= static_cast<int64_t>(seq_id)) {
        ++batch;
    }
    const int64_t pos_id = pos_ids[seq_id];
    const int64_t block_id = block_ids[batch * table_width + pos_id / block_size + 1]; // +1 跳过每行的 count 前缀
    const int64_t token_id = pos_id % block_size;
    const size_t in_offset = seq_id * numel;
    const size_t out_offset = (block_id * block_size + token_id) * numel;
    T *out = _out + out_offset;
    const T *in = _in + in_offset;
    for(size_t i = tid; i < numel; i += blockDim.x){
        out[i] = in[i];
    }
}


template <typename T>
void kv_cache_move_launch(std::byte *out,       
                          const std::byte *in,       
                          const std::byte *block_ids,
                          const std::byte *cut_idx, 
                          const std::byte *pos_ids,  
                          const size_t block_size,    
                          const size_t tot_token_num, 
                          const size_t table_width,
                          const size_t numel)
{
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_in = reinterpret_cast<const T *>(in);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_pos_ids = reinterpret_cast<const int64_t *>(pos_ids);

    dim3 blockDim(256);
    dim3 gridDim(tot_token_num);

    kv_cache_move_kernel<<<gridDim, blockDim>>>(d_out, d_in, d_block_ids, d_cut_idx, d_pos_ids, block_size, table_width, numel);

    CUDA_CHECK(cudaGetLastError());
}

// 量化写入: per-(token, nkvh) 行 absmax => scale => absmax/127 => round(fp/scale)
template <typename T>
__global__ void kv_cache_move_quant_kernel(int8_t *__restrict__ _out,
                                           float *__restrict__ _scale,
                                           const T *__restrict__ _in,
                                           const int64_t *__restrict__ block_ids,
                                           const int64_t *__restrict__ cut_idx,
                                           const int64_t *__restrict__ pos_ids,
                                           const size_t block_size,
                                           const size_t table_width,
                                           const size_t nkvh,
                                           const size_t dh)
{
    const size_t seq_id = blockIdx.x;   // 全局 token 偏移
    const size_t tid = threadIdx.x;
    // 反查所属 batch：cut_idx[batch] <= seq_id < cut_idx[batch+1]
    size_t batch = 0;
    while (cut_idx[batch + 1] <= static_cast<int64_t>(seq_id)) {
        ++batch;
    }
    const int64_t pos_id = pos_ids[seq_id];
    const int64_t block_id = block_ids[batch * table_width + pos_id / block_size + 1]; // +1 跳过每行的 count 前缀
    const int64_t token_id = pos_id % block_size;

    const T *in = _in + seq_id * nkvh * dh;
    int8_t *out = _out + (block_id * block_size + token_id) * nkvh * dh;
    float *scale = _scale + (block_id * block_size + token_id) * nkvh;

    __shared__ float warp_max[32];
    __shared__ float row_scale;
    const int lane = static_cast<int>(tid) & 31;
    const int warp = static_cast<int>(tid) >> 5;

    for (size_t h = 0; h < nkvh; ++h) {
        float local_max = 0.f;
        for (size_t j = tid; j < dh; j += blockDim.x) {
            local_max = fmaxf(local_max, fabsf(llaisys::utils::nvidia::cast<float>(in[h * dh + j])));
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffffu, local_max, off));
        }
        if (lane == 0) {
            warp_max[warp] = local_max;
        }
        __syncthreads();
        float amax = 0.f;
        if (tid < (blockDim.x + 31) / 32) {
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
            scale[h] = row_scale;
        }
        __syncthreads();
        const float inv = 1.f / row_scale;
        for (size_t j = tid; j < dh; j += blockDim.x) {
            const float q = nearbyintf(llaisys::utils::nvidia::cast<float>(in[h * dh + j]) * inv);
            out[h * dh + j] = static_cast<int8_t>(max(-128, min(127, static_cast<int>(q))));
        }
        __syncthreads();   // 下一个 h 复用 warp_max 前同步
    }
}

template <typename T>
void kv_cache_move_quant_launch(std::byte *out,
                                std::byte *scale,
                                const std::byte *in,
                                const std::byte *block_ids,
                                const std::byte *cut_idx,
                                const std::byte *pos_ids,
                                const size_t block_size,
                                const size_t tot_token_num,
                                const size_t table_width,
                                const size_t nkvh,
                                const size_t dh)
{
    dim3 blockDim(256);
    dim3 gridDim(tot_token_num);

    kv_cache_move_quant_kernel<T><<<gridDim, blockDim>>>(
        reinterpret_cast<int8_t *>(out), reinterpret_cast<float *>(scale), reinterpret_cast<const T *>(in),
        reinterpret_cast<const int64_t *>(block_ids), reinterpret_cast<const int64_t *>(cut_idx),
        reinterpret_cast<const int64_t *>(pos_ids), block_size, table_width, nkvh, dh);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void kv_cache_move(std::byte *out,       
                     const std::byte *in,       
                     const std::byte *block_ids,
                     const std::byte *cut_idx, 
                     const std::byte *pos_ids,  
                     std::byte *scale,
                     const size_t block_size,    
                     const size_t tot_token_num, 
                     const size_t table_width,
                     llaisysDataType_t out_dtype,
                     llaisysDataType_t in_dtype,
                     const size_t numel,
                     const size_t nkvh)
{
    // 量化写入路径：out 为 I8，scale 输出 F32 per-(token,nkvh)
    if (out_dtype == LLAISYS_DTYPE_I8) {
        const size_t dh = numel / nkvh;
        switch (in_dtype) {
        case LLAISYS_DTYPE_F32:
            return kv_cache_move_quant_launch<float>(out, scale, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, nkvh, dh);
        case LLAISYS_DTYPE_BF16:
            return kv_cache_move_quant_launch<llaisys::bf16_t>(out, scale, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, nkvh, dh);
        case LLAISYS_DTYPE_F16:
            return kv_cache_move_quant_launch<llaisys::fp16_t>(out, scale, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, nkvh, dh);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(in_dtype);
        }
    }
    // 原样搬移路径
    switch (out_dtype) {
    case LLAISYS_DTYPE_F32:
        return kv_cache_move_launch<float>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    case LLAISYS_DTYPE_BF16:
        return kv_cache_move_launch<llaisys::bf16_t>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    case LLAISYS_DTYPE_F16:
        return kv_cache_move_launch<llaisys::fp16_t>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(out_dtype);
    }
}
} // namespace llaisys::ops::nvidia
