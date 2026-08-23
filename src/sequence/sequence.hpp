#pragma once
#include "../utils.hpp"
#include "../config.hpp"
#include "../core/llaisys_core.hpp"

#include <vector>
#include <memory>
#include <condition_variable>
#include <mutex>
#include <random>

namespace llaisys{

class Sequence;
using seq_t = std::shared_ptr<Sequence>;

enum class SeqStatus {
    WAITING = 0, // 等待prefill
    RUNNING = 1, // 进行decode
    FINISHED = 2 // 生成结束
};

class Sequence{
private:
    size_t _seq_id;             // 序列号 其实没用
    SeqStatus _status;          // 当前状态 (等待prefill、decode中、结束)
    bool _is_prefill;           // 是否处于prefill阶段
    long long _prompt_hash;     // prompt部分hash值，根据这个hash值来确定请求是否匹配
    size_t _ntoken;             // 总token数
    size_t _cached_ntoken;      // 已缓存(计算)token数
    size_t _prompt_ntoken;      // prompt部分总token数
    size_t _scheduled_ntoken;   // 规划计算token数
    size_t _max_ntoken;         // 最大产生token数
    int64_t _last_token;        // 最后一个token信息
    std::vector<int64_t> _token_ids;    // token列表
    std::vector<int64_t> _block_ids;    // 缓存块表
    // 异步等待 + 取消
    std::mutex _token_mutex;
    std::condition_variable _token_cv;
    bool _cancelled;
    // per-request 采样参数 (构造时固化)
    int _top_k;
    float _top_p;
    float _temperature;
    std::mt19937 _rng;

public:
    Sequence(int64_t *token_ids, size_t ntoken, size_t max_ntoken = MAX_TOKEN_NUM,
             int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);

    ~Sequence() = default;

    // Prevent copy
    Sequence(const Sequence &) = delete;
    Sequence &operator=(const Sequence &) = delete;

    // Prevent move
    Sequence(Sequence &&) = delete;
    Sequence &operator=(Sequence &&) = delete;

    size_t seq_id();
    SeqStatus &status();
    bool &is_prefill();
    bool is_finish();          
    long long prompt_hash();

    size_t token_num();
    size_t block_num();
    size_t &cached_token_num();
    size_t prompt_token_num();
    size_t max_token_num();
    size_t &scheduled_token_num();
    size_t last_block_token_num();

    int64_t last_token();
    int64_t token_at(size_t index);
    std::vector<int64_t> &token_ids();
    std::vector<int64_t> &block_ids();

    std::pair<int64_t *, size_t> block(size_t i);
    void add(int64_t token);

    // 异步等待 (condvar, 不空转)
    bool wait_for_token(size_t ntoken);
    void notify_token();
    // 取消
    void cancel();
    bool cancelled();
    // 采样参数
    int top_k();
    float top_p();
    float temperature();
    std::mt19937 &rng();

    // seq_id 计数器
    static size_t counter(){
        static size_t count = 0;
        return ++ count;
    }
};
};