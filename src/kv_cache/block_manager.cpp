#include "block_manager.hpp"

#include <algorithm>
#include <cstring>

namespace llaisys{

BlockManager::BlockManager(const size_t token_dim, 
                           const size_t token_num, 
                           const size_t block_num,
                           llaisysDataType_t dtype, 
                           llaisysDeviceType_t device_type,
                           int device_id): 
                        _token_num(token_num),
                        _block_num(block_num),
                        _dtype(dtype),
                        _device_type(device_type),
                        _device_id(device_id)
{
    // 初始化块信息
    for(size_t i = 0; i < block_num; ++ i){
        Block block{static_cast<int>(i), -1, std::vector<int64_t>(token_num, -1)};
        this->_blocks.push_back(std::make_shared<Block>(block));
        this->_free_block_ids.push(static_cast<int>(i));
    }
    // 申请底部缓存张量
    tensor_t _cache = Tensor::create({2, block_num * token_num * token_dim}, dtype, device_type, device_id);
    this->_k_cache = _cache->slice(0, 0, 1);
    this->_v_cache = _cache->slice(0, 1, 2);
}

BlockManager_t BlockManager::create(const size_t token_dim, 
                                    const size_t token_num, 
                                    const size_t block_num,
                                    llaisysDataType_t dtype, 
                                    llaisysDeviceType_t device_type,
                                    int device_id)
{
    static BlockManager_t manager = BlockManager_t(new BlockManager(token_dim, token_num, block_num, dtype, device_type, device_id));
    return manager;
}

static constexpr long long Prime = 133331;
static constexpr long long mod = 998244353;

long long BlockManager::_compute_hash(const int64_t *token_ids, size_t l, size_t r, long long prefix)
{
    ASSERT(l <= r, "BlockManager_compute_hash: l > r.");
    long long hash = prefix == -1 ? 0 : prefix;
    for(size_t i = l; i <= r; ++ i){
        hash = (hash * Prime + token_ids[i]) % mod;
    }
    return hash;
}

void BlockManager::_reset(block_t block){
    block->_hash = -1;
    std::fill(block->_token_ids.begin(), block->_token_ids.end(), -1);
}

void BlockManager::_update(block_t block, long long hash, const int64_t *token_ids, size_t l, size_t r){
    ASSERT(l <= r, "BlockManager_update: l > r.");
    ASSERT(block->_token_ids.size() >= r - l + 1, "BlockManager_update: too many.");
    block->_hash = hash;
    std::copy(token_ids + l, token_ids + r + 1, block->_token_ids.begin());
}

size_t BlockManager::allocate(const int64_t *token_ids, 
                            const size_t ntoken, 
                            std::vector<int> &block_ids)
{
    const size_t token_num = this->_token_num;
    const size_t nblock = (ntoken + token_num - 1) / token_num;
    size_t need = nblock; // 需要的块数
    ASSERT(this->_block_num >= need, "BlockManager_allocate: need more than have.");
    // 前缀匹配 找可利用的缓存
    size_t start = ntoken;
    long long prefix = -1;
    block_ids = std::vector<int>(nblock, 0);
    for(size_t i = 0; i < nblock; ++ i){
        size_t l = i * token_num;
        size_t r = std::min((i + 1) * token_num - 1, ntoken - 1);
        long long block_hash = this->_compute_hash(token_ids, l, r, prefix);
        if(!this->_hash_2_block.count(block_hash)){
            start = l;
            break;
        }
        -- need, prefix = block_hash;
        block_ids[i] = this->_hash_2_block[block_hash];
    }
    if(need == 1){
        long long hash = prefix;
        for(size_t i = start; i < ntoken; ++ i){
            hash = this->_compute_hash(token_ids, i, i, hash);
            if(this->_hash_2_block.count(hash)){
                start = i + 1;
                block_ids[nblock - 1] = this->_hash_2_block[hash];
            }
        }
        if(start > (nblock - 1) * token_num){
            need = 0;
            int64_t block_id = block_ids[nblock - 1];
            size_t l = (nblock - 1) * token_num;
            size_t r = ntoken - 1;
            long long block_hash = this->_compute_hash(token_ids, l, r, prefix);
            this->_hash_2_block.erase(this->_blocks[block_id]->_hash);
            this->_hash_2_block[block_hash] = block_id;
            this->_update(this->_blocks[block_id], block_hash, token_ids, l, r);
        }
    }
    // 剩余的不够 从最近分配释放
    while(this->_free_block_ids.size() < need){
        ASSERT(!this->_last_used_block.empty(), "BlockManager_allocate: cannot free enough blocks.");
        size_t last = this->_last_used_block.top();
        this->_last_used_block.pop();
        this->_used_block_ids.erase(last);
        this->_free_block_ids.push(last);
        this->_hash_2_block.erase(this->_blocks[last]->_hash);
        this->_reset(this->_blocks[last]);
    }
    // 占用新分配的block
    for(size_t i = nblock - need; i < nblock; ++ i){
        int64_t block_id = this->_free_block_ids.front();
        this->_free_block_ids.pop();
        size_t l = i * token_num;
        size_t r = std::min((i + 1) * token_num - 1, ntoken - 1);
        long long block_hash = this->_compute_hash(token_ids, l, r, prefix);
        this->_used_block_ids.insert(block_id);
        this->_last_used_block.push(block_id);
        this->_hash_2_block[block_hash] = block_id;
        this->_update(this->_blocks[block_id], block_hash, token_ids, l, r);
        block_ids[i] = block_id;
        prefix = block_hash;
    }
    return start;
}

};