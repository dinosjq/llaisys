#include "scheduler.hpp"
#include <algorithm>

namespace llaisys{

Scheduler::Scheduler(size_t token_num, 
            size_t block_num, 
            size_t batch_max_token_num, 
            size_t batch_max_seq_num, 
            int64_t eos):
            _cur(0),
            _token_num(token_num),
            _batch_max_token_num(batch_max_token_num),
            _batch_max_seq_num(batch_max_seq_num),
            _eos(eos)
{   
    this->_manager = std::make_shared<BlockManager>(token_num, block_num);
}

void Scheduler::add(seq_t seq) { 
    std::unique_lock<std::mutex> lock(this->_lock);
    this->_waiting[!this->_cur].push(seq);
    lock.unlock();
}

std::pair<std::vector<seq_t>, bool> Scheduler::schedule(){
    std::unique_lock<std::mutex> lock(this->_lock);
    int cur_id = _cur;
    lock.unlock();
    if(this->_waiting[cur_id].empty()){
        lock.lock();
        cur_id = !cur_id;
        this->_cur = cur_id;
        lock.unlock();
    }
    std::vector<seq_t> scheduled_seqs;
    size_t batch_token_num = 0;
    while(!this->_waiting[cur_id].empty() && scheduled_seqs.size() < this->_batch_max_seq_num){
        seq_t seq = this->_waiting[cur_id].front();
        size_t remaining = this->_batch_max_token_num - batch_token_num;
        (void)remaining;
        if(remaining == 0){
            break;
        }
        size_t token_num = 0;
        int cached_block_num = 0;
        if(seq->block_ids().empty()){
            cached_block_num = this->_manager->can_allocate(seq);
            if(cached_block_num == -1){
                break;
            }
            token_num = seq->token_num() - cached_block_num * this->_token_num;
        }else{
            token_num = seq->token_num() - seq->cached_token_num();
        }
        if(remaining < token_num && !this->_waiting[cur_id].empty()){ // 对首个超限的做chunked prefill
            break;
        }
        if(seq->block_ids().empty()){
            this->_manager->allocate(seq, cached_block_num);
        }
        seq->scheduled_token_num() = std::min(token_num, remaining); // chunked prefill 
        batch_token_num += seq->scheduled_token_num();
        if(seq->cached_token_num() + seq->scheduled_token_num() == seq->token_num()){
            this->_waiting[cur_id].pop();
            this->_running.push_back(seq);
            seq->status() = SeqStatus::RUNNING;
        }
        scheduled_seqs.push_back(seq);
    }
    if(!scheduled_seqs.empty()){
        return std::make_pair(scheduled_seqs, true);
    }
    while(!this->_running.empty() && scheduled_seqs.size() < this->_batch_max_seq_num){
        seq_t seq = this->_running.front();
        this->_running.pop_front();
        size_t remaining = this->_batch_max_token_num - batch_token_num;
        (void)remaining;
        while(!this->_manager->can_append(seq)){
            if(!this->_running.empty()){
                this->preempty(this->_running.front());
                this->_running.pop_front();
            }else{
                this->preempty(seq);
                break;
            }
        }
        if(!this->_manager->can_append(seq)){
            break;
        }
        seq->scheduled_token_num() = 1;
        seq->is_prefill() = false;
        this->_manager->may_append(seq);
        scheduled_seqs.push_back(seq);
    }
    for(seq_t seq: scheduled_seqs) this->_running.push_back(seq);
    return std::make_pair(scheduled_seqs, false);
}

void Scheduler::preempty(seq_t &seq){
    this->_manager->deallocate(seq);
    seq->cached_token_num() = 0;
    seq->is_prefill() = true;
    seq->status() = SeqStatus::WAITING;
    this->add(seq);
}

void Scheduler::postprocess(std::vector<seq_t> &seqs, std::vector<int64_t> token_ids, bool is_prefill){
    const size_t seq_num = seqs.size();
    for(size_t i = 0; i < seq_num; ++ i){
        seq_t seq = seqs[i];
        int64_t token_id = token_ids[i];
        this->_manager->hash_blocks(seq);
        seq->cached_token_num() += seq->scheduled_token_num();
        seq->scheduled_token_num() = 0;
        if(is_prefill && seq->cached_token_num() < seq->token_num()){
            continue;
        }
        seq->add(token_id);
        if(token_id == this->_eos || seq->token_num() >= seq->max_token_num()){
            seq->status() = SeqStatus::FINISHED;
            this->_manager->deallocate(seq);
            auto it = std::find(this->_running.begin(), this->_running.end(), seq);
            this->_running.erase(it);
        }
    }
}

};