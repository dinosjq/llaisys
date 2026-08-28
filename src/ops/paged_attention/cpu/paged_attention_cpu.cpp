#include "paged_attention_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
void paged_attention(std::byte *attn_val,       // 输出 (tot_seqlen, nh, dv)
                     const std::byte *q,        // q (tot_seqlen, nh, d)
                     const std::byte *k_cache,  // k底层缓存 (tot_len, nkvh, d) 
                     const std::byte *v_cache,  // q底层缓存 (tot_len, nkvh, dv)
                     const std::byte *block_ids,// 块表 (batch, block_ids) 记录块号 (block_num, block_size, nkvh, d | dv) 
                     const std::byte *cut_idx,  // 记录对应序列起始位置，也可计算seqlen
                     const std::byte *tot_len,  // 已计算kv总长
                     const size_t block_size,   // 每个块的token数
                     const size_t batch_size,   // 批次大小
                     const size_t max_block_num,// 块表最多块数
                     const size_t tot_task_num, // 任务总数 Σceil(seq_len/BLOCK_M)
                     llaisysDataType_t dtype,   // 权重数据类型
                     const float &scale,        // 缩放因子
                     const size_t &nh,          // 其他参数 ...
                     const size_t &dv, 
                     const size_t &d, 
                     const size_t &nkvh){
	(void)attn_val;
	(void)q;
	(void)k_cache;
	(void)v_cache;
	(void)block_ids;
	(void)cut_idx;
	(void)tot_len;
	(void)block_size;
	(void)batch_size;
	(void)max_block_num;
	(void)tot_task_num;
	(void)dtype;
	(void)scale;
	(void)nh;
	(void)dv;
	(void)d;
	(void)nkvh;
	TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::cpu
