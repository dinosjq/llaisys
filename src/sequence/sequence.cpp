#include "sequence.hpp"
#include "../kv_cache/block_manager.hpp"

namespace llaisys{
Sequence::Sequence(int64_t *token_ids, size_t ntoken, size_t max_ntoken,
                   int top_k, float top_p, float temperature):
                    _seq_id(Sequence::counter()),
                    _status(SeqStatus::WAITING),
                    _is_prefill(true),
                    _prompt_hash(BlockManager::_compute_hash(token_ids, ntoken, -1)),
                    _ntoken(ntoken),
                    _cached_ntoken(0),
                    _prompt_ntoken(ntoken),
                    _scheduled_ntoken(0),
                    _max_ntoken(max_ntoken),
                    _last_token(token_ids[ntoken - 1]),
                    _cancelled(false),
                    _top_k(top_k),
                    _top_p(top_p),
                    _temperature(temperature)
{
    this->_token_ids = std::vector<int64_t>(token_ids, token_ids + ntoken);
    this->_block_ids = std::vector<int64_t>();
}

size_t Sequence::seq_id(){
    return this->_seq_id;
}

SeqStatus &Sequence::status(){
    return this->_status;
}

bool &Sequence::is_prefill(){
    return this->_is_prefill;
}

bool Sequence::is_finish(){ 
    return this->_status == SeqStatus::FINISHED; 
}
long long Sequence::prompt_hash(){
    return this->_prompt_hash;
}

size_t Sequence::token_num(){
    return this->_ntoken;
}

size_t Sequence::block_num(){ 
    return (this->_ntoken + KV_CACHE_TOKEN_NUM  - 1) / KV_CACHE_TOKEN_NUM; 
}

size_t &Sequence::cached_token_num(){
    return this->_cached_ntoken;
}

size_t Sequence::prompt_token_num(){
    return this->_prompt_ntoken;
}

size_t Sequence::max_token_num(){
    return this->_max_ntoken;
}

size_t &Sequence::scheduled_token_num(){
    return this->_scheduled_ntoken;
}

size_t Sequence::last_block_token_num() { 
    return this->_ntoken - (this->block_num() - 1) * KV_CACHE_TOKEN_NUM; 
}

int64_t Sequence::last_token(){
    return this->_last_token;
}

std::vector<int64_t> &Sequence::token_ids(){
    return this->_token_ids;
}

std::vector<int64_t> &Sequence::block_ids(){
    return this->_block_ids;
}

std::pair<int64_t *, size_t> Sequence::block(size_t i){ 
    ASSERT(0 <= i && i < this->block_num(), "Sequence_block: out range.");
    size_t block_token_num = (i != this->block_num() - 1) ? KV_CACHE_TOKEN_NUM : this->last_block_token_num();
    return std::make_pair(this->_token_ids.data() + i * KV_CACHE_TOKEN_NUM, block_token_num);
}

void Sequence::add(int64_t token){
    {
        std::lock_guard<std::mutex> lock(this->_token_mutex);
        this->_token_ids.push_back(token);
        this->_last_token = token;
        ++ this->_ntoken;
    }
    // 唤醒等待新 token 的请求线程
    this->_token_cv.notify_all();
}

// condvar 真等待: 直到 token 数超过 ntoken 或被取消
// 返回 true=有新token false=已取消
bool Sequence::wait_for_token(size_t ntoken) {
    std::unique_lock<std::mutex> lock(this->_token_mutex);
    this->_token_cv.wait(lock, [&]{
        return this->_cancelled || this->_ntoken > ntoken;
    });
    return !this->_cancelled;
}

void Sequence::notify_token() {
    std::unique_lock<std::mutex> lock(this->_token_mutex);
    lock.unlock();
    this->_token_cv.notify_all();
}

void Sequence::cancel() {
    {
        std::lock_guard<std::mutex> lock(this->_token_mutex);
        this->_cancelled = true;
    }
    this->_token_cv.notify_all();
}

bool Sequence::cancelled() {
    std::lock_guard<std::mutex> lock(this->_token_mutex);
    return this->_cancelled;
}

int Sequence::top_k() { return this->_top_k; }
float Sequence::top_p() { return this->_top_p; }
float Sequence::temperature() { return this->_temperature; }

};