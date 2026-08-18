#pragma once

#include "runtime/model_context.hpp"
#include "../sequence/sequence.hpp"
#include "../scheduler/scheduler.hpp"
#include "../config.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace llaisys::model {

struct ModelRequest {
    seq_t sequence;
    size_t observed_tokens;
    std::mutex mutex;
};
using model_request_t = std::shared_ptr<ModelRequest>;

class Model {
protected:
    // 设备
    llaisysDeviceType_t _device_type = LLAISYS_DEVICE_CPU;
    std::vector<int> _device_ids;
    // 请求调度、kv cache
    scheduler_t _scheduler;
    std::vector<tensor_t> _k_cache;
    std::vector<tensor_t> _v_cache;
    std::mutex _request_lock;
    std::vector<model_request_t> _active_requests;
    // 事件循环
    bool _running = false;
    std::thread _worker;
    // 分层上下文（由子类构造具体派生类型）
    std::unique_ptr<ModelContext> _ctx;
    bool _meta_ready = false;

    // 子类实现：组装 Layer → logits → 采样返回 token
    virtual std::vector<int64_t> forward(BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) = 0;
    // 子类实现：绑定 KV 缓存
    virtual void bind_kv_caches() = 0;
    // 子类实现：绑定上下文运行时信息
    virtual void bind_context_runtime(BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) = 0;

public:
    virtual ~Model();

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    Model() = default;

    // Loader surface (C API byte map → subclass typed readers via utils::as / as_string)
    virtual void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) = 0;
    virtual void set_weight(const std::string &hf_name, tensor_t tensor) = 0;

    bool meta_ready() const { return this->_meta_ready; }

    void start();
    void stop();

    // 提交请求
    model_request_t submit(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                           int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);
    // 等待请求完成
    int64_t await(const model_request_t &request);
    // 取消请求
    void abort(const model_request_t &request);
    // 释放请求
    void release(const model_request_t &request);

    // 预处理：准备分层表、预填充、解码
    static std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs, size_t block_table_width);
    static BatchPack prepare_prefill(const std::vector<seq_t> &seqs);
    static BatchPack prepare_decode(const std::vector<seq_t> &seqs);

    ModelContext &ctx() { return *_ctx; }
    const ModelContext &ctx() const { return *_ctx; }
};

} // namespace llaisys::model
