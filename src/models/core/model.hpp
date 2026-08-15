#pragma once

#include "model_context.hpp"
#include "../../sequence/sequence.hpp"
#include "../../scheduler/scheduler.hpp"
#include "../../config.hpp"

#include <cstddef>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace llaisys::core {

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
    // 分层上下文
    ModelContext _ctx;

    // 子类实现：组装 Layer → logits → 采样返回 token
    virtual std::vector<int64_t> forward(BatchPack &pack, std::vector<int64_t> &block_ids, bool is_prefill) = 0;
    // 权重已由 SetWeight 写入 _ctx；worker 启动前只把 KV 视图绑到 Context
    virtual void bind_kv_caches() {}

public:
    virtual ~Model();

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    Model() = default;

    // 事件循环
    void start();
    void stop();

    // 请求生命周期
    model_request_t submit(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                           int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);
    int64_t await(const model_request_t &request);
    void abort(const model_request_t &request);
    void release(const model_request_t &request);

    // 上传边界：WeightRole → Context 分层槽
    static void set_weight(ModelContext &ctx, WeightRole role, size_t layer, tensor_t tensor);

    // 预处理
    static std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs, size_t block_table_width);
    static BatchPack prepare_prefill(const std::vector<seq_t> &seqs);
    static BatchPack prepare_decode(const std::vector<seq_t> &seqs);

    ModelContext &ctx() { return _ctx; }
    const ModelContext &ctx() const { return _ctx; }
};

} // namespace llaisys::core
