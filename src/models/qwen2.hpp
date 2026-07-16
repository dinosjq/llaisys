#pragma once

#include "../../include/llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../kv_cache/block_manager.hpp"
#include "../scheduler/scheduler.hpp"
#include <mutex>
#include <shared_mutex>
#include <thread>

namespace llaisys{

class Qwen2;
class Sequence;
using Qwen2_t = std::shared_ptr<Qwen2>;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

struct Qwen2Pack {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx;
    std::vector<int64_t> tot_len;
    std::vector<float> top_p;       // per-seq 采样参数
    std::vector<float> temperature; // per-seq 采样参数
    size_t max_seq_len;
};

using Qwen2Weights = ::LlaisysQwen2Weights;

struct Qwen2Tensors{
    tensor_t _x_norm;
    tensor_t _q;
    tensor_t _k;
    tensor_t _v;
    tensor_t _q_rope;
    tensor_t _k_rope;
    tensor_t _attn_val;
    tensor_t _attn_out;
    tensor_t _x_attn;
    tensor_t _m_norm;
    tensor_t _gate;
    tensor_t _up;
    tensor_t _swiglu;
    tensor_t _down;
    tensor_t _x_mlp;
    tensor_t _dev_block_ids;
    tensor_t _dev_cut_idx;
    tensor_t _dev_tot_len;
    tensor_t _dev_token_ids;
    tensor_t _dev_pos_ids;
    tensor_t _x;
    tensor_t _x_part;
    tensor_t _x_part_norm;
    tensor_t _logits;
    tensor_t _top_idx;
    tensor_t _top_val;
};

class Qwen2{
private:
    // 模型、设备信息
    Qwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    std::vector<int> _device_ids;
    Qwen2Weights _weights;
    Qwen2Tensors _tensors;
    // 请求调度、kv cache管理
    scheduler_t _scheduler;
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    std::unordered_map<long long, std::shared_ptr<Sequence>> _hash_2_seq;
    std::shared_mutex _hash_lock;
    // 事件循环
    bool _running;
    bool _profiled = false;
    std::thread _worker;
    // logprobs: 保存最近一次 forward 的 logits 快照 (CPU 独立副本, batch x voc)
    // worker 线程写入 / 请求线程读取, 由 _logits_mutex 保护
    tensor_t _last_logits;
    std::mutex _logits_mutex;

public:
    Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    ~Qwen2();

    // Prevent copy
    Qwen2(const Qwen2 &) = delete;
    Qwen2 &operator=(const Qwen2 &) = delete;

    // Prevent move
    Qwen2(Qwen2 &&) = delete;
    Qwen2 &operator=(Qwen2 &&) = delete;

    // 事件循环
    void start();
    // 终止系统
    void stop();

    // 请求
    int64_t request(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                    int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);

    // 取消请求
    void abort(int64_t *token_ids, size_t ntoken);

    // 获取上一步 logprobs (从 _last_logits 按 batch_idx 切片)
    int get_logprobs(float *out_logprobs, int n_vocab,
                     int *out_tokens, int n_tokens, int batch_idx);

    // 填充块表
    std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs);

    // 预处理
    Qwen2Pack prepare_prefill(const std::vector<seq_t> &seqs);
    Qwen2Pack prepare_decode(const std::vector<seq_t> &seqs);

    // 批次前向传播
    std::vector<int64_t> forward(Qwen2Pack &pack, std::vector<int64_t> &block_ids);    

    // 模型权重
    Qwen2Weights &weights(){ return this->_weights; }
};

};