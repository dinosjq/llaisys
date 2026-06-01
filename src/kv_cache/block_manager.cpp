#include "block_manager.hpp"

#include <algorithm>
#include <cstring>

namespace{
template<typename T>
bool check_equal(const T *it_0, const T *it_1, const size_t n) {
    for(size_t i = 0; i < n; ++ i){
        if(it_0[i] != it_1[1]) return false;
    }
    return true;
}
};

namespace llaisys{

BlockManager::BlockManager(const size_t token_num, 
                           const size_t block_num): 
                        _token_num(token_num),
                        _block_num(block_num)
{
    // 初始化块信息
    for(size_t i = 0; i < block_num; ++ i){
        Block block{static_cast<int>(i), -1, std::vector<int64_t>(token_num, -1)};
        this->_blocks.push_back(std::make_shared<Block>(block));
        this->_free_block_ids.push_back(static_cast<int>(i));
    }
}

// 自定义哈希计算函数
long long BlockManager::_compute_hash(const int64_t *token_ids, size_t n, long long prefix) {
    ASSERT(n >= 0, "BlockManager_compute_hash: n < 0.");
    long long hash = prefix == -1 ? 0 : prefix;
    for(size_t i = 0; i < n; ++ i){
        hash = (hash * Prime + token_ids[i]) % mod;
    }
    return hash;
}

// 分配一个块 并清空块上的数据
int64_t BlockManager::_allocate_block(){
    ASSERT(!this->_free_block_ids.empty(), "BlockManager_allocate_block: free_queue is empty.");
    int64_t block_id = this->_free_block_ids.front();
    this->_free_block_ids.pop_front();
    block_t block = this->_blocks[block_id];
    ASSERT(block->_ref_count == 0, "BlockManager_allocate_block: ref_count not 0");
    if(block->_hash != -1 && this->_hash_2_block[block->_hash] == block->_block_id){
        this->_hash_2_block.erase(block->_hash);
    }
    block->_reset();
    this->_used_block_ids.insert(block_id);
    return block_id;
}
    
// 释放所有权 再次使用需要重新申请 不清空块上数据
void BlockManager::_deallocate_block(int64_t block_id){
    ASSERT(block_id >= 0 && size_t(block_id) < this->_block_num, "BlockManager_deallocate_block: out range.");
    ASSERT(this->_blocks[block_id]->_ref_count == 0, "BlockManager_deallocate_block: ref_count not 0.");
    this->_used_block_ids.erase(block_id);
    this->_free_block_ids.push_back(block_id);
}

// 对 prefill 的 seq 判断 kv cache 是否足够分配
int BlockManager::can_allocate(seq_t seq){
    // 前缀哈希
    long long hash = -1;
    // prompt 部分预计占用block数 已命中缓存块数(hash一致 且used) 需要新分配块数(hash未命中 或者 hash命中,但是所有权释放,没有used)
    const size_t block_num = seq->block_num(); 
    int cached_block_num = 0;
    int new_block_num = block_num;
    // 只对整块进行匹配
    for(size_t i = 0; i < block_num - 1; ++ i){
        const auto &[token_ids, ntoken] = seq->block(i);
        hash = this->_compute_hash(token_ids, ntoken, hash);
        // hash 未命中
        if(!this->_hash_2_block.count(hash)){ 
            break; 
        }
        // 命中 检查一致性 预防 hash 冲突
        const int block_id = this->_hash_2_block[hash];
        if(!check_equal(this->_blocks[block_id]->_token_ids.data(), token_ids, ntoken)){
            break;
        }
        // 已命中缓存块数增加
        ++ cached_block_num;
        // 查看是否 used
        if(this->_used_block_ids.count(block_id)){
            -- new_block_num; // hash命中并且used 需要新分配块数减少
        }
    }
    // 如果剩余的块数小于需要新分配的部分 就返回-1 不够
    if(this->_free_block_ids.size() < size_t(new_block_num)) return -1;
    // 否则返回命中的前缀块数
    return cached_block_num;
}

// 对 prefill 的 seq 进行 kv cache 分配 cached_block_num 为已匹配前缀块数
void BlockManager::allocate(seq_t seq, size_t cached_block_num){
    std::vector<int64_t> &block_ids = seq->block_ids();
    ASSERT(block_ids.empty(), "BlockManager_allocate: block_ids not empty.");
    // 根据前缀哈希 先分配已经匹配的部分
    long long hash = -1;
    for(size_t i = 0; i < cached_block_num; ++ i){
        const auto &[token_ids, ntoken] = seq->block(i);
        hash = this->_compute_hash(token_ids, ntoken, hash);
        const int64_t block_id = this->_hash_2_block[hash];
        // 已经被使用的 更新引用计数
        if(this->_used_block_ids.count(block_id)){
            ++ this->_blocks[block_id]->_ref_count;
        }else{
            // 被释放过所有权的 更新引用计数为一 从free中清除 然后加入used
            this->_blocks[block_id]->_ref_count = 1;
            auto it = std::find(this->_free_block_ids.begin(), this->_free_block_ids.end(), block_id);
            this->_free_block_ids.erase(it);
            this->_used_block_ids.insert(block_id);
        }
        block_ids.push_back(block_id);
    }
    // 分配剩下没有命中的部分
    const size_t block_num = seq->block_num();
    for(size_t i = cached_block_num; i < block_num; ++ i){
        block_ids.push_back(this->_allocate_block());
    }
    // 更新缓存信息
    seq->cached_token_num() = cached_block_num * this->_token_num;
}

// 释放 seq 占据 kv cache 块的所有权
void BlockManager::deallocate(seq_t seq){
    // 得到对应的块表
    std::vector<int64_t> &block_ids = seq->block_ids();
    for(int i = int(block_ids.size() - 1); i >= 0; -- i){
        int64_t block_id = block_ids[i];
        block_t block = this->_blocks[block_id];
        // 更新引用计数
        -- block->_ref_count;
        if(block->_ref_count == 0){
            // 引用计数为 0 时才能被其他序列占用 数据依旧没有真正释放
            this->_deallocate_block(block_id);
        }
    }
    // 更新序列信息
    seq->cached_token_num() = 0;
    block_ids.clear();
}

// 判断是否存在需要分配但是不够的情况
bool BlockManager::can_append(seq_t seq){
    return this->_free_block_ids.size() >= (seq->token_num() % this->_token_num == 1);
}

// 只有新增 token 占据新块时采取申请
void BlockManager::may_append(seq_t seq){
    if(seq->token_num() % this->_token_num == 1){
        seq->block_ids().push_back(this->_allocate_block());
    }
}

// 更新序列所用块的信息 
void BlockManager::hash_blocks(seq_t seq){
    // 先计算对应位置的开始与结束的块表索引 
    // 这里 decode 阶段凑齐整块才会更新一次 应该是减少更新频率 还有就是因为对于kv cache的分配是按块分配的 只有整块才参与分配
    size_t start = seq->cached_token_num() / this->_token_num;
    size_t end = (seq->cached_token_num() + seq->scheduled_token_num()) / this->_token_num;
    if(start == end) return;
    // 计算前缀哈希
    long long hash = (start > 0) ? _compute_hash(seq->token_ids().data(), (start - 1) * this->_token_num, -1) : -1;
    // 按块进行更新
    for(int i = start; i < (int)end; ++ i){
        // 查找块表并找到对应块
        int block_id = seq->block_ids()[i];
        block_t block = this->_blocks[block_id];
        // 从序列中取出对应的token_id信息和长度信息
        const auto &[token_ids, ntoken] = seq->block(i);
        // 计算 hash 值更新到对应块上
        hash = _compute_hash(token_ids, ntoken, hash);
        block->_update(hash, token_ids, ntoken);
        // 更新 hash 映射
        this->_hash_2_block[hash] = block->_block_id;
    }
}   

};