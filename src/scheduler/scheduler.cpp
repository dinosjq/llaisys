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
    // 初始化kv cache管理器
    this->_manager = std::make_shared<BlockManager>(token_num, block_num);
}

// 上锁原子加到另一个缓冲队列
void Scheduler::add(seq_t seq) { 
    std::unique_lock<std::mutex> lock(this->_lock);
    this->_waiting[!this->_cur].push(seq);
    lock.unlock();
}

// 执行调度
std::pair<std::vector<seq_t>, bool> Scheduler::schedule(){
    // 先查看当前缓冲区的等待队列是否有请求，否则转到另一个
    std::unique_lock<std::mutex> lock(this->_lock);
    int cur_id = _cur;
    lock.unlock();
    if(this->_waiting[cur_id].empty()){
        lock.lock();
        cur_id = !cur_id;
        this->_cur = cur_id;
        lock.unlock();
    }
    // 本次选中的序列 和 当前批次选中序列要计算的token数
    std::vector<seq_t> scheduled_seqs;
    size_t batch_token_num = 0;
    size_t remaining = 0;
    // 先找待 prefill 的部分
    while(!this->_waiting[cur_id].empty() && scheduled_seqs.size() < this->_batch_max_seq_num){
        // 选一个
        seq_t seq = this->_waiting[cur_id].front();
        // 被取消的序列: 释放资源并跳过
        if(seq->cancelled()){
            this->_waiting[cur_id].pop();
            seq->status() = SeqStatus::FINISHED;
            this->_manager->deallocate(seq);
            continue;
        }
        // 计算还剩多少
        remaining = this->_batch_max_token_num - batch_token_num;
        if(remaining == 0){
            break;
        }
        size_t token_num = 0;
        int cached_block_num = 0;
        // 如果当前没分配过kv cache
        if(seq->block_ids().empty()){
            // 先看都不够分配 不够就 break
            cached_block_num = this->_manager->can_allocate(seq);
            if(cached_block_num == -1){
                break;
            }
            // 够分配就计算 prompt 待计算的 token 数
            token_num = seq->token_num() - cached_block_num * this->_token_num;
        }else{
            token_num = seq->token_num() - seq->cached_token_num();
        }
        // 没分配的就正式进行分配
        if(seq->block_ids().empty()){
            this->_manager->allocate(seq, cached_block_num);
        }
        // chunked prefill 更新序列信息 和 批次计算信息
        seq->scheduled_token_num() = std::min(token_num, remaining); // chunked prefill 
        batch_token_num += seq->scheduled_token_num();
        // 如果本次计算完了 prompt 部分 就将序列加入 decode 的队列
        if(seq->cached_token_num() + seq->scheduled_token_num() == seq->token_num()){
            this->_waiting[cur_id].pop();
            this->_running.push_back(seq);
            seq->status() = SeqStatus::RUNNING;
        }
        // 正式加入调度队列
        scheduled_seqs.push_back(seq);
    }
    // 如果存在待 prefill 的 就返回调度队列和 is_prefill=true 
    if(!scheduled_seqs.empty()){
        return std::make_pair(scheduled_seqs, true);
    }
    // 如果没有待 prefill 的了 就调度 decode 的
    while(!this->_running.empty() && scheduled_seqs.size() < this->_batch_max_seq_num){
        seq_t seq = this->_running.front();
        this->_running.pop_front();
        // 被取消的序列: 释放资源并跳过
        if(seq->cancelled()){
            seq->status() = SeqStatus::FINISHED;
            this->_manager->deallocate(seq);
            continue;
        }
        remaining = this->_batch_max_token_num - batch_token_num;
        if(remaining == 0){
            break;
        }
        // 与上面类似 只不过这里 seq 一定分配过 kv cache
        // 只有当新增 token 需要占有新的 block 时才需要再分配一块 block
        // 需要新分配 block 但是剩余缓存不足时
        while(!this->_manager->can_append(seq)){
            // 先尝试将其他请求占有的 kv cache 进行释放
            if(!this->_running.empty()){
                this->preempty(this->_running.front());
                this->_running.pop_front();
            }else{
                // 全释放完了也不行 就将当前的请求占有的kv cache释放
                this->preempty(seq);
                break;
            }
        }
        // 再次判断是否需要分配 需要时能不能分配
        if(!this->_manager->can_append(seq)){
            break;
        }
        // 更新序列信息
        seq->scheduled_token_num() = 1;
        seq->is_prefill() = false;
        this->_manager->may_append(seq);
        scheduled_seqs.push_back(seq);
    }
    // 重新加入调度队列
    for(seq_t seq: scheduled_seqs) this->_running.push_back(seq);
    // 返回调度队列 和 prefill=false
    return std::make_pair(scheduled_seqs, false);
}

// 预清空缓存信息 实际上是释放所有权 并没有完全释放块上信息 只有再次分配给序列时 才做清除
void Scheduler::preempty(seq_t &seq){
    // 使用kv cache manager释放
    this->_manager->deallocate(seq);
    // 更新序列信息
    seq->cached_token_num() = 0;
    seq->is_prefill() = true;
    seq->status() = SeqStatus::WAITING;
    // 重新加入到等待队列
    this->add(seq);
}

// 取消请求
// 线程安全设计: _waiting/_running/BlockManager 均只由 worker 线程(schedule/postprocess)修改,
// abort 由请求线程调用, 因此这里只标记 cancelled 并唤醒等待线程,
// 队列清理与 kv cache 释放由 worker 线程在 schedule() 弹出时检测完成
void Scheduler::abort(seq_t seq) {
    seq->cancel();
}

// 后处理更新对应的序列占有的 kv cache 缓存信息
void Scheduler::postprocess(std::vector<seq_t> &seqs, std::vector<int64_t> token_ids, bool is_prefill){
    const size_t seq_num = seqs.size();
    for(size_t i = 0; i < seq_num; ++ i){
        seq_t seq = seqs[i];
        int64_t token_id = token_ids[i];
        // 更新缓存信息
        this->_manager->hash_blocks(seq);
        // 更新序列信息
        seq->cached_token_num() += seq->scheduled_token_num();
        seq->scheduled_token_num() = 0;
        // prefill 不需要新增
        if(is_prefill && seq->cached_token_num() < seq->token_num()){
            continue;
        }
        // 新增 token
        seq->add(token_id);
        // 查看是否到达结束状态 
        if(token_id == this->_eos || seq->token_num() >= seq->max_token_num()){
            // 更新信息
            seq->status() = SeqStatus::FINISHED;
            // 释放kv cache所有权
            this->_manager->deallocate(seq);
            // 从调度队列中清除
            auto it = std::find(this->_running.begin(), this->_running.end(), seq);
            this->_running.erase(it);
        }
    }
}

};