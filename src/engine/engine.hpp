#pragma once

#include <cstddef>
#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../sequence/sequence.hpp"
#include "../scheduler/scheduler.hpp"
#include "../config.hpp"

// 前向声明
namespace llaisys::model {
class Model;
}

namespace llaisys::engine {

// 一批次的输入打包
struct BatchPack {
    std::vector<int64_t> token_ids;
    std::vector<int64_t> pos_ids;
    std::vector<int64_t> cut_idx;
    std::vector<int64_t> tot_len;
    std::vector<int64_t> block_ids;
    std::vector<int> top_ks;
    std::vector<float> top_ps;
    std::vector<float> temperatures;
    std::vector<std::mt19937 *> rngs;
    size_t batch_size = 0;
    size_t tot_task_num = 0;
    bool is_prefill = true;
};

using BatchInput = std::variant<BatchPack>; // 后续可添加 context 信息作为输入
using BatchOutput = std::variant<std::vector<int64_t>>;

struct Request {
    seq_t sequence;
    size_t observed_ntoken;
    std::mutex mutex;
};
using request_t = std::shared_ptr<Request>;

struct ParallelConfig {
    size_t tp;
    size_t pp;
};

// Worker 前向声明
class Worker;
using worker_t = std::unique_ptr<Worker>;

class Engine {
private:
    // 设备
    llaisysDeviceType_t _device_type = LLAISYS_DEVICE_CPU;
    int _device_id;
    // 调度
    scheduler_t _scheduler;
    std::mutex _request_lock;
    std::mutex _start_lock;  // 保护 start()/事件循环启动，避免并发 submit 竞争
    std::vector<request_t> _active_requests;
    // 执行
    worker_t _worker;
    // 事件循环
    bool _running = false;
    std::thread _runner;

public:
    Engine(llaisysDeviceType_t device, int device_id, std::unique_ptr<model::Model> model);
    ~Engine();

    Engine(const Engine &) = delete;
    Engine &operator=(const Engine &) = delete;
    Engine(Engine &&) = delete;
    Engine &operator=(Engine &&) = delete;

    void start();
    void stop();

    // 提交请求
    request_t submit(int64_t *token_ids, size_t ntoken, int64_t max_new_ntoken,
                     int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);
    // 等待请求完成
    int64_t await(const request_t &request);
    // 取消请求
    void abort(const request_t &request);
    // 释放请求
    void release(const request_t &request);

    // 模型访问（C API 加载权重入口）
    model::Model &model();
    Worker &worker();

private:
    void _ensure_scheduler();
};

} // namespace llaisys::engine
