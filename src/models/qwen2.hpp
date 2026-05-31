#pragma once

#include "../../include/llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../kv_cache/block_manager.hpp"
#include "../scheduler/scheduler.hpp"
#include <mutex>
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
    std::vector<int> cut_idx;
    std::vector<int> tot_len;
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
    std::mutex _hash_lock;
    // 事件循环
    bool _running;
    std::thread _worker;

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
    int64_t request(int64_t *token_ids, size_t ntoken);

    // 填充块表
    std::vector<int> prepare_block_table(const std::vector<seq_t> &seqs);

    // 预处理
    Qwen2Pack prepare_prefill(const std::vector<seq_t> &seqs);
    Qwen2Pack prepare_decode(const std::vector<seq_t> &seqs);

    // 批次前向传播
    std::vector<int64_t> forward(Qwen2Pack &pack, std::vector<int> &block_ids);    

    // 模型权重
    Qwen2Weights &weights(){ return this->_weights; }
};

};