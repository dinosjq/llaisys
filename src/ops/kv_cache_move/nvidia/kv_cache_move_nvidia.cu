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
                                    const size_t token_num,    
                                    const size_t max_block_num,
                                    const size_t numel)
{
    const size_t batch = blockIdx.y;
    const size_t offset = blockIdx.x;
    const size_t tid = threadIdx.x;
    const size_t lo = cut_idx[batch];
    const size_t up = cut_idx[batch + 1] - 1;
    const size_t seqlen = up - lo + 1;
    if(offset >= seqlen) return;
    const size_t seq_id = lo + offset;
    const int64_t pos_id = pos_ids[seq_id];
    const int64_t block_id = block_ids[batch * max_block_num + pos_id / token_num + 1]; // +1 跳过每行的 count 前缀
    const int64_t token_id = pos_id % token_num;
    const size_t in_offset = seq_id * numel;
    const size_t out_offset = (block_id * token_num + token_id) * numel;
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
                          const size_t token_num,    
                          const size_t batch_size,  
                          const size_t max_block_num,
                          const size_t max_seq_len, 
                          const size_t numel)
{
    auto *d_out = reinterpret_cast<T *>(out);
    const auto *d_in = reinterpret_cast<const T *>(in);
    const auto *d_block_ids = reinterpret_cast<const int64_t *>(block_ids);
    const auto *d_cut_idx = reinterpret_cast<const int64_t *>(cut_idx);
    const auto *d_pos_ids = reinterpret_cast<const int64_t *>(pos_ids);

    dim3 blockDim(256);
    dim3 gridDim(max_seq_len, batch_size);

    kv_cache_move_kernel<<<gridDim, blockDim>>>(d_out, d_in, d_block_ids, d_cut_idx, d_pos_ids, token_num, max_block_num, numel);

    CUDA_CHECK(cudaGetLastError());
}

namespace llaisys::ops::nvidia {
void kv_cache_move(std::byte *out,       
                     const std::byte *in,       
                     const std::byte *block_ids,
                     const std::byte *cut_idx, 
                     const std::byte *pos_ids,  
                     const size_t token_num,    
                     const size_t batch_size,  
                     const size_t max_block_num,
                     const size_t max_seq_len, 
                     llaisysDataType_t dtype, 
                     const size_t numel)
{
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return kv_cache_move_launch<float>(out, in, block_ids, cut_idx, pos_ids, token_num, batch_size, max_block_num, max_seq_len, numel);
    case LLAISYS_DTYPE_BF16:
        return kv_cache_move_launch<llaisys::bf16_t>(out, in, block_ids, cut_idx, pos_ids, token_num, batch_size, max_block_num, max_seq_len, numel);
    case LLAISYS_DTYPE_F16:
        return kv_cache_move_launch<llaisys::fp16_t>(out, in, block_ids, cut_idx, pos_ids, token_num, batch_size, max_block_num, max_seq_len, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(dtype);
    }
}
} // namespace llaisys::ops::nvidia
