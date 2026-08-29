#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
// out_dtype == I8 时走量化写入（in 为 FP，scale 输出 per-(token,nkvh) F32），否则原样搬移（scale=nullptr）。
void kv_cache_move(std::byte *out,       
                     const std::byte *in,       
                     const std::byte *block_ids,
                     const std::byte *cut_idx, 
                     const std::byte *pos_ids,  
                     std::byte *scale,
                     const size_t block_size,    // 每块 token 数
                     const size_t tot_token_num, // 本批次待搬移 token 总数
                     const size_t table_width,
                     llaisysDataType_t out_dtype,
                     llaisysDataType_t in_dtype,
                     const size_t numel,         // nkvh * dh
                     const size_t nkvh);
};
