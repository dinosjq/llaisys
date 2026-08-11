#pragma once
#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::nvidia {
void flash_decoding(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv) => (batch_size, nh, dv)
                    std::byte *attn_acc,       // 输出 (batch_block_num: sum{[[block_num_0], [block_num_1], ...]}, nh, dv) 
                    std::byte *attn_sum,       // 输出 (batch_block_num, nh, 1)
                    std::byte *attn_max,       // 输出 (batch_block_num, nh, 1)
                    const std::byte *q,        // q (tot_seqlen, nh, d) => (batch_size, nh, d)
                    const std::byte *k_cache,  // k底层缓存 (block_num * token_num, nkvh, d)  
                    const std::byte *v_cache,  // q底层缓存 (block_num * token_num, nkvh, dv)
                    const std::byte *block_ids,// 块表 (batch, block_ids_size | block_ids)  kv cache布局: (block_num, token_num, nkvh, d | dv) 
                    const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                    const std::byte *tot_len,  // 已计算kv总长
                    const size_t token_num,    // 每个块的token数 (批次无关)
                    const size_t batch_size,   // 批次大小 (批次有关)
                    const size_t max_block_num,// 单个序列的块表最多块数 (批次无关)
                    const size_t tot_block_num,// 整个批次所有序列所用块数
                    const size_t max_seq_len,  // 最大序列长度 用于分块 (批次无关)
                    llaisysDataType_t dtype,   // 权重数据类型
                    const float &scale,        // 缩放因子
                    const size_t &nh,          // 其他参数 ...
                    const size_t &dv, 
                    const size_t &d, 
                    const size_t &nkvh);
};