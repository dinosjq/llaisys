#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void paged_attention(std::byte *attn_val,       // 输出 (seqlen, nh, dv)
                     const std::byte *q,        // q (seqlen, nh, d)
                     const std::byte *k_cache,  // k底层缓存 (totlen, nkvh, d)
                     const std::byte *v_cache,  // q底层缓存 (totlen, nkvh, dv)
                     const std::byte *block_ids,// 块表 记录块号 (block_num, token_num, nkvh, d | dv) 
                     const size_t block_num,    // 总块数
                     const size_t token_num,    // 每个块的token数
                     const size_t nblock,       // 块表中块的个数
                     llaisysDataType_t dtype,   // 权重数据类型
                     const float &scale,        // 缩放因子
                     const size_t &seqlen,      // 需计算序列长度
                     const size_t &nh,          // 其他参数 ...
                     const size_t &dv, 
                     const size_t &d, 
                     const size_t &totlen, 
                     const size_t &nkvh);
}