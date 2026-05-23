#include "paged_attention_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {
void paged_attention(std::byte *attn_val,
					 const std::byte *q,
					 const std::byte *k_cache,
					 const std::byte *v_cache,
					 const std::byte *block_ids,
					 const size_t block_num,
					 const size_t token_num,
					 const size_t nblock,
					 llaisysDataType_t dtype,
					 const float &scale,
					 const size_t &seqlen,
					 const size_t &nh,
					 const size_t &dv,
					 const size_t &d,
					 const size_t &totlen,
					 const size_t &nkvh) {
	(void)attn_val;
	(void)q;
	(void)k_cache;
	(void)v_cache;
	(void)block_ids;
	(void)block_num;
	(void)token_num;
	(void)nblock;
	(void)dtype;
	(void)scale;
	(void)seqlen;
	(void)nh;
	(void)dv;
	(void)d;
	(void)totlen;
	(void)nkvh;
	TO_BE_IMPLEMENTED();
}
} // namespace llaisys::ops::cpu
