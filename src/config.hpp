// 一些参数配置
#include <math.h>

// kv cache
#define KV_CACHE_TOKEN_NUM 64   // kv cache块上token数
#define KV_CACHE_BLOCK_NUM 128  // kv cache总块数

// sample
#define TOP_K 10    // 采样top_k
#define TOP_P 0.9   // 采样累计概率前top_p

// bacth limit
#define BATCH_MAX_TOKEN_NUM 8192    // 批次最大并行token总数
#define BATCH_MAX_SEQ_NUM 64        // 批次最大并行seq总数

// seq limit
#define MAX_TOKEN_NUM 2048          // 序列最大长度上界

// block table
#define MAX_BLOCK_NUM (std::min((MAX_TOKEN_NUM + KV_CACHE_TOKEN_NUM - 1) / KV_CACHE_TOKEN_NUM, KV_CACHE_BLOCK_NUM) + 1)  // 单个序列最大块数
 
#define BATCH_MAX_BLOCK_NUM int(1.0 * (BATCH_MAX_TOKEN_NUM + BATCH_MAX_SEQ_NUM * (KV_CACHE_TOKEN_NUM - 1)) / KV_CACHE_TOKEN_NUM)

