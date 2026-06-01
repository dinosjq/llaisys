#pragma once
#include "llaisys.h"

#include <cstddef>

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
                     const size_t numel);
};
