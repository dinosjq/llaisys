// 一些参数配置

// kv cache
#define KV_CACHE_TOKEN_NUM 64   // kv cache块上token数
#define KV_CACHE_BLOCK_NUM 128  // kv cache总块数

// sample
#define TOP_K 10    // 采样top_k
#define TOP_P 0.9   // 采样累计概率前top_p

// bacth limit
#define BATCH_MAX_TOKEN_NUM 16386   // 批次最大并行token总数
#define BATCH_MAX_SEQ_NUM 64        // 批次最大并行seq总数

// seq limit
#define MAX_TOKEN_NUM 16386  // 序列最大长度上界

