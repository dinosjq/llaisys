#pragma once

#include "../utils.hpp"
#include "../tensor/tensor.hpp"
#include "../sequence/sequence.hpp"
#include "../kv_cache/block_manager.hpp"
#include "../config.hpp"
#include "../core/llaisys_core.hpp"

#include <queue>
#include <unordered_map>
#include <mutex>
#include <list>

namespace llaisys{

class Scheduler;
using scheduler_t = std::shared_ptr<Scheduler>;

class Scheduler{
private:
    // 调度队列 + 双缓冲
    std::mutex _lock;
    volatile int _cur;
    std::queue<seq_t> _waiting[2];
    std::list<seq_t> _running;
    // kv cache 调度
    size_t _token_num;
    BlockManager_t _manager;
    // 其他 limit
    size_t _batch_max_token_num;
    size_t _batch_max_seq_num;
    // 结束符
    int64_t _eos;

public:
    Scheduler(size_t token_num, 
              size_t block_num, 
              size_t batch_max_token_num, 
              size_t batch_max_seq_num, 
              int64_t eos);

    ~Scheduler() = default;

    // Prevent copy
    Scheduler(const Scheduler &) = delete;
    Scheduler &operator=(const Scheduler &) = delete;

    // Prevent move
    Scheduler(Scheduler &&) = delete;
    Scheduler &operator=(Scheduler &&) = delete;

    // 原子添加请求信息到未使用的等待队列
    void add(seq_t seq);

    // 执行 PD 分离调度 [本次计算序列, is_prefill]
    std::pair<std::vector<seq_t>, bool> schedule();
    
    // 预先清空对应使用的kv cache信息
    void preempty(seq_t &seq);

    // 后处理更新对应缓存块的信息
    void postprocess(std::vector<seq_t> &seqs, std::vector<int64_t> token_ids, bool is_prefill);
    
    // 每块 token 数
    size_t token_num() { return this->_token_num; }
};
};