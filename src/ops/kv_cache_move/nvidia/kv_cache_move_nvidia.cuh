#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void kv_cache_move(std::byte *out,       
                     const std::byte *in,       
                     const std::byte *block_ids,
                     const std::byte *cut_idx, 
                     const std::byte *pos_ids,  
                     const size_t block_size,    // 每块 token 数
                     const size_t tot_token_num, // 本批次待搬移 token 总数
                     const size_t table_width,
                     llaisysDataType_t dtype, 
                     const size_t numel);
};
