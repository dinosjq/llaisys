#pragma once
#include "../tensor/tensor.hpp"
#include "../core/llaisys_core.hpp"
#include "../sequence/sequence.hpp"

#include <memory>
#include <unordered_map>
#include <vector>
#include <atomic>
#include <queue>
#include <stack>
#include <unordered_set>
#include <list>

namespace llaisys{

static constexpr long long Prime = 133331;
static constexpr long long mod = 998244353;

struct Block;
using block_t = std::shared_ptr<Block>;

class BlockManager;
using BlockManager_t = std::shared_ptr<BlockManager>;
 
// kv cache block
struct Block{
    int64_t _block_id;                  // 块号
    long long _hash = -1;               // 哈希值 前缀+当前块信息
    std::vector<int64_t> _token_ids;    // 当前块缓存的token信息
    size_t _ref_count = 0;              // 引用计数
    
    // 重新初始化block
    void _reset(){
        _hash = -1; 
        _ref_count = 1;
        std::fill(_token_ids.begin(), _token_ids.end(), -1);
    }

    // 更新block
    void _update(long long hash, const int64_t *token_ids, size_t n){
        ASSERT(_token_ids.size() >= n, "Block_update: too many.");
        _hash = hash;
        std::copy(token_ids, token_ids + n, _token_ids.begin());
    }
};

// kv cache block manager
class BlockManager{
private:
    size_t _token_num;                                      // 单个block所包含token缓存个数
    size_t _block_num;                                      // 维护的总kv cache块数
    std::vector<block_t> _blocks;                           // 维护所有块的信息
    std::unordered_map<long long, int64_t> _hash_2_block;   // 哈希值转块号
    std::list<int64_t> _free_block_ids;                     // 可用的block的块号
    std::unordered_set<int64_t> _used_block_ids;            // 已经使用的block的块号

private:
    // 分配一个块
    int64_t _allocate_block();

    // 释放一个块
    void _deallocate_block(int64_t block_id);

public:
    BlockManager(const size_t token_num, const size_t block_num); 

    ~BlockManager() = default;

    // Prevent copy
    BlockManager(const BlockManager &) = delete;
    BlockManager &operator=(const BlockManager &) = delete;

    // Prevent move
    BlockManager(BlockManager &&) = delete;
    BlockManager &operator=(BlockManager &&) = delete;
    
    // 自定义哈希函数求解: 计算 token_ids[0, n) 及其前缀信息 prefix 的哈希值
    static long long _compute_hash(const int64_t *token_ids, size_t n, long long prefix = -1);

    // 判断能否分配 能就返回已缓存块数 否则返回-1
    int can_allocate(seq_t seq);
    
    // 给对应seq进行分配
    void allocate(seq_t seq, size_t cached_block_num);

    // 释放对应seq占有的kv cache块
    void deallocate(seq_t seq);

    // 当需要新块时 判断能否新占有
    bool can_append(seq_t seq);

    // 按需要分配一块
    void may_append(seq_t);

    // 更新所用块的信息
    void hash_blocks(seq_t seqs);
};
}