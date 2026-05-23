#pragma once
#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <queue>
#include <stack>
#include <unordered_set>

namespace llaisys{

struct Block;
using block_t = std::shared_ptr<Block>;

class BlockManager;
using BlockManager_t = std::shared_ptr<BlockManager>;
 
// kv cache block
struct Block{
    int _block_id;                          // 块号
    long long _hash = -1;                   // 哈希值 前缀+当前块信息
    std::vector<int64_t> _token_ids;        // 当前块缓存的token信息
    // std::atomic<size_t> _ref_count = 0;
};

// kv cache block manager
class BlockManager{
private:
    tensor_t _k_cache;                                      // 底层连续缓存张量
    tensor_t _v_cache;
    size_t _token_num;                                      // 单个block所包含token缓存个数
    size_t _block_num;                                      // 维护的总kv cache块数
    std::vector<block_t> _blocks;                           // 维护所有块的信息
    std::unordered_map<long long, int> _hash_2_block;       // 哈希值转块号
    std::queue<int> _free_block_ids;                        // 可用的block的块号
    std::unordered_set<int> _used_block_ids;                // 已经使用的block的块号
    std::stack<int> _last_used_block;                       // TODO: 后续实现为 LRU cache 
    llaisysDataType_t _dtype;                               // 缓存数据类型
    llaisysDeviceType_t _device_type;                       // 缓存设备类型
    int _device_id;                                         // 缓存所在设备

private:
    // 自定义哈希函数求解: 计算 token_ids[l: r + 1) 及其前缀信息 prefix 的哈希值
    long long _compute_hash(const int64_t *token_ids, size_t l, size_t r, long long prefix = -1);

    // 重新初始化block
    void _reset(block_t block);

    // 更新block
    void _update(block_t block, long long hash, const int64_t *token_ids, size_t l, size_t r);

    
    BlockManager(const size_t token_dim,        // 单个token的k/v cache占用dtype个数
                 const size_t token_num = 32,   // 单个block所包含token总数 
                 const size_t block_num = 8,    // block总数
                 llaisysDataType_t dtype= LLAISYS_DTYPE_F32,                // cache的数据类型
                 llaisysDeviceType_t device_type = LLAISYS_DEVICE_NVIDIA,   // cache所处设备类型
                 int device_id = 0);                                        // cache所处设备号

public:
    static BlockManager_t create(const size_t token_dim,       
                 const size_t token_num = 128,  
                 const size_t block_num = 8, 
                 llaisysDataType_t dtype= LLAISYS_DTYPE_F32,               
                 llaisysDeviceType_t device_type = LLAISYS_DEVICE_NVIDIA, 
                 int device_id = 0); 

    ~BlockManager() = default;

    // Prevent copy
    BlockManager(const BlockManager &) = delete;
    BlockManager &operator=(const BlockManager &) = delete;

    // Prevent move
    BlockManager(BlockManager &&) = delete;
    BlockManager &operator=(BlockManager &&) = delete;
    
    // 根据token_ids和现有缓存信息分配block块表（对应设备上） 返回匹配前缀块总长
    size_t allocate(const int64_t *token_ids,       // 要分配的序列的信息
                  const size_t ntoken,              // 序列总长
                  std::vector<int> &block_ids);     // 分配的 block 信息 块表/总块数 (输出)
                 

    // 根据id获取对应的块的指针
    block_t block(int block_id) { 
        ASSERT(block_id >= 0 && static_cast<size_t>(block_id) < this->_block_num, "BlockManager_block: block_id out range.");
        return this->_blocks[block_id]; 
    }
    
    // 获取底层缓存的张量
    tensor_t k_cache() { return this->_k_cache; };
    tensor_t v_cache() { return this->_v_cache; };

    // 返回总块数
    size_t block_num() { return this->_block_num; }
    // 返回单个block所占token数
    size_t token_num() { return this->_token_num; }
};
}