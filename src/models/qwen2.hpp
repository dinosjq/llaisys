#pragma once

#include "../../include/llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../kv_cache/block_manager.hpp"
#include "../scheduler/scheduler.hpp"
#include "framework/model_context.hpp"
#include "qwen2_pack.hpp"
#include <mutex>
#include <random>
#include <thread>

namespace llaisys{

class Qwen2;
class Sequence;
using Qwen2_t = std::shared_ptr<Qwen2>;

struct Qwen2Request {
    seq_t sequence;
    size_t observed_tokens;
    std::mutex mutex;
};
using qwen2_request_t = std::shared_ptr<Qwen2Request>;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
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
    tensor_t _attn_acc;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, dh) F32
    tensor_t _attn_sum;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, 1)  F32
    tensor_t _attn_max;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, 1)  F32
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
    std::mutex _request_lock;
    std::vector<qwen2_request_t> _active_requests;
    // 事件循环
    bool _running;
    std::thread _worker;
    // 分层框架上下文（权重/KV/workspace 绑定；默认走 layer stack）
    framework::ModelContext _ctx;
    bool use_layer_forward_ = true;
    // logprobs 下次迭代实现: 添加 mutex + 需要时 forward 中保存 logits C PU 快照

    // 方案 A：从 legacy Weights 经 WeightRole/set_weight 同步到 ModelContext（加载结束 / start 前）
    void sync_weights_from_legacy_struct();
    void bind_context_runtime(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    // 与 legacy 相同的 topk + sample 后处理
    std::vector<int64_t> sample_from_logits(tensor_t logits, Qwen2Pack &pack);

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

    // 提交独立请求：每个 request 持有自己的 Sequence，KV Cache 复用由 BlockManager 管理
    qwen2_request_t submit(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                           int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);
    // 等待当前 request 生成下一个 token
    int64_t await(const qwen2_request_t &request);
    // 取消当前 request，不影响相同 prompt 的其他请求
    void abort(const qwen2_request_t &request);
    // 释放 request；未完成的 Sequence 会先从调度器取消
    void release(const qwen2_request_t &request);

    // 填充块表
    std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs);

    // 预处理
    Qwen2Pack prepare_prefill(const std::vector<seq_t> &seqs);
    Qwen2Pack prepare_decode(const std::vector<seq_t> &seqs);

    // 批次前向传播（按 use_layer_forward_ 分发）
    std::vector<int64_t> forward(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    // 硬编码循环（回滚路径：LLAISYS_QWEN2_LAYER_FORWARD=0）
    std::vector<int64_t> forward_legacy(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill);
    // Layer 组装路径（默认）：填 Context → run_qwen2_layer_stack → 同 legacy 采样
    std::vector<int64_t> forward_layers(Qwen2Pack &pack, std::vector<int64_t> &block_ids, bool is_prefill);

    bool use_layer_forward() const { return use_layer_forward_; }
    void set_use_layer_forward(bool enabled) { use_layer_forward_ = enabled; }
    // 最近一次 forward 写入的 logits 视图（供双路径对照）
    tensor_t last_logits(size_t batch_size) const;

    // 模型权重
    Qwen2Weights &weights(){ return this->_weights; }
};

};