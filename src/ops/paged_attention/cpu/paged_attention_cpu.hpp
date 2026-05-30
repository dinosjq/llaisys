#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void paged_attention(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                     const std::byte *q,        // q (tot_seqlen, nh, d)
                     const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                     const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                     const std::byte *block_ids,// 块表 (batch, block_ids) 记录块号 (block_num, token_num, nkvh, d | dv) 
                     const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                     const std::byte *tot_len,  // 已计算kv总长
                     const size_t token_num,    // 每个块的token数
                     const size_t batch_size,   // 批次大小
                     const size_t max_block_num,// 块表最多块数
                     const size_t max_seq_len,  // 最大序列长度 用于分块
                     llaisysDataType_t dtype,   // 权重数据类型
                     const float &scale,        // 缩放因子
                     const size_t &nh,          // 其他参数 ...
                     const size_t &dv, 
                     const size_t &d, 
                     const size_t &nkvh);
};