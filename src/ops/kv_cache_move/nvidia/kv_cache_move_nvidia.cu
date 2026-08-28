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

namespace llaisys::ops::nvidia {
void kv_cache_move(std::byte *out,       
                     const std::byte *in,       
                     const std::byte *block_ids,
                     const std::byte *cut_idx, 
                     const std::byte *pos_ids,  
                     const size_t block_size,    
                     const size_t tot_token_num, 
                     const size_t table_width,
                     llaisysDataType_t dtype, 
                     const size_t numel)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return kv_cache_move_launch<float>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    case LLAISYS_DTYPE_BF16:
        return kv_cache_move_launch<llaisys::bf16_t>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    case LLAISYS_DTYPE_F16:
        return kv_cache_move_launch<llaisys::fp16_t>(out, in, block_ids, cut_idx, pos_ids, block_size, tot_token_num, table_width, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
