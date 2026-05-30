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

long long BlockManager::_compute_hash(const int64_t *token_ids, size_t n, long long prefix) {
    ASSERT(n >= 0, "BlockManager_compute_hash: n < 0.");
    long long hash = prefix == -1 ? 0 : prefix;
    for(size_t i = 0; i < n; ++ i){
        hash = (hash * Prime + token_ids[i]) % mod;
    }
    return hash;
}

int BlockManager::_allocate_block(){
    ASSERT(!this->_free_block_ids.empty(), "BlockManager_allocate_block: free_queue is empty.");
    int block_id = this->_free_block_ids.front();
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
    
void BlockManager::_deallocate_block(int block_id){
    ASSERT(block_id >= 0 && size_t(block_id) < this->_block_num, "BlockManager_deallocate_block: out range.");
    ASSERT(this->_blocks[block_id]->_ref_count == 0, "BlockManager_deallocate_block: ref_count not 0.");
    this->_used_block_ids.erase(block_id);
    this->_free_block_ids.push_back(block_id);
}

int BlockManager::can_allocate(seq_t seq){
    long long hash = -1;
    const size_t block_num = seq->block_num();
    int cached_block_num = 0;
    int new_block_num = block_num;
    for(size_t i = 0; i < block_num - 1; ++ i){
        const auto &[token_ids, ntoken] = seq->block(i);
        hash = this->_compute_hash(token_ids, ntoken, hash);
        if(!this->_hash_2_block.count(hash)){ 
            break; 
        }
        const int block_id = this->_hash_2_block[hash];
        if(!check_equal(this->_blocks[block_id]->_token_ids.data(), token_ids, ntoken)){
            break;
        }
        ++ cached_block_num;
        if(this->_used_block_ids.count(block_id)){
            -- new_block_num;
        }
    }
    if(this->_free_block_ids.size() < size_t(new_block_num)) return -1;
    return cached_block_num;
}

void BlockManager::allocate(seq_t seq, size_t cached_block_num){
    std::vector<int> &block_ids = seq->block_ids();
    ASSERT(block_ids.empty(), "BlockManager_allocate: block_ids not empty.");
    long long hash = -1;
    for(size_t i = 0; i < cached_block_num; ++ i){
        const auto &[token_ids, ntoken] = seq->block(i);
        hash = this->_compute_hash(token_ids, ntoken, hash);
        const int block_id = this->_hash_2_block[hash];
        if(this->_used_block_ids.count(block_id)){
            ++ this->_blocks[block_id]->_ref_count;
        }else{
            this->_blocks[block_id]->_ref_count = 1;
            auto it = std::find(this->_free_block_ids.begin(), this->_free_block_ids.end(), block_id);
            this->_free_block_ids.erase(it);
            this->_used_block_ids.insert(block_id);
        }
        block_ids.push_back(block_id);
    }
    const size_t block_num = seq->block_num();
    for(size_t i = cached_block_num; i < block_num; ++ i){
        block_ids.push_back(this->_allocate_block());
    }
    seq->cached_token_num() = cached_block_num * this->_token_num;
}

void BlockManager::deallocate(seq_t seq){
    std::vector<int> &block_ids = seq->block_ids();
    for(int i = int(block_ids.size() - 1); i >= 0; -- i){
        int block_id = block_ids[i];
        block_t block = this->_blocks[block_id];
        -- block->_ref_count;
        if(block->_ref_count == 0){
            this->_deallocate_block(block_id);
        }
    }
    seq->cached_token_num() = 0;
    block_ids.clear();
}

bool BlockManager::can_append(seq_t seq){
    return this->_free_block_ids.size() >= (seq->token_num() % this->_token_num == 1);
}

void BlockManager::may_append(seq_t seq){
    if(seq->token_num() % this->_token_num == 1){
        seq->block_ids().push_back(this->_allocate_block());
    }
}

void BlockManager::hash_blocks(seq_t seq){
    size_t start = seq->cached_token_num() / this->_token_num;
    size_t end = (seq->cached_token_num() + seq->scheduled_token_num()) / this->_token_num;
    if(start == end) return;
    long long hash = (start > 0) ? _compute_hash(seq->token_ids().data(), (start - 1) * this->_token_num, -1) : -1;
    for(int i = start; i < (int)end; ++ i){
        int block_id = seq->block_ids()[i];
        block_t block = this->_blocks[block_id];
        const auto &[token_ids, ntoken] = seq->block(i);
        hash = _compute_hash(token_ids, ntoken, hash);
        block->_update(hash, token_ids, ntoken);
        this->_hash_2_block[hash] = block->_block_id;
    }
}   

};