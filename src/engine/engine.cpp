#include "engine.hpp"

#include "worker/worker.hpp"

namespace llaisys::engine {

Engine::Engine(llaisysDeviceType_t device, int device_id, std::unique_ptr<model::Model> model)
    : _device_type(device), _device_id(device_id),
      _worker(std::make_unique<Worker>(device, device_id, std::move(model))) {}

model::Model &Engine::model() {
    return this->_worker->model();
}

Worker &Engine::worker() {
    return *this->_worker;
}

Engine::~Engine() {
    this->stop();
    if (this->_runner.joinable()) {
        this->_runner.join();
    }
}

void Engine::_ensure_scheduler() {
    if (!this->_scheduler) {
        this->_scheduler = std::make_shared<Scheduler>(KV_CACHE_BLOCK_SIZE, KV_CACHE_BLOCK_NUM, BATCH_MAX_TOKEN_NUM,
                                                       BATCH_MAX_SEQ_NUM, this->_worker->model().end_token());
    }
}

void Engine::start() {
    std::lock_guard<std::mutex> lock(this->_start_lock);
    if (this->_running) {
        return;
    }
    // 首次启动时创建调度器（需要模型 end_token）
    this->_ensure_scheduler();

    // 事件循环
    auto _loop = [](Engine *engine) -> void {
        while (engine->_running) {
            // 调度
            auto [seqs, is_prefill] = engine->_scheduler->schedule();
            if (seqs.empty())
                continue;
            // 预处理
            BatchInput input = engine->_worker->prepare(seqs, MAX_BLOCK_NUM, is_prefill);
            // 前向
            BatchOutput output = engine->_worker->forward(input);
            auto token_ids = std::get<std::vector<int64_t>>(output);
            // 后处理
            engine->_scheduler->postprocess(seqs, token_ids, is_prefill);
        }
    };
    // Genshen启动!
    this->_running = true;
    this->_runner = std::thread(_loop, this);
}

void Engine::stop() {
    this->_running = false;
}

request_t Engine::submit(int64_t *token_ids, size_t ntoken, int64_t max_new_ntoken, int top_k, float top_p, float temperature) {
    if (!this->_worker->is_ready()) {
        throw std::runtime_error("model meta is not ready; call set_meta before submit");
    }
    // 首次提交时再 sync + 启动 worker
    this->start();
    size_t limit = (max_new_ntoken < 0) ? MAX_TOKEN_NUM : ntoken + static_cast<size_t>(max_new_ntoken);
    auto request = std::make_shared<Request>();
    request->sequence = std::make_shared<Sequence>(token_ids, ntoken, limit, top_k, top_p, temperature);
    request->observed_ntoken = ntoken;
    {
        std::lock_guard<std::mutex> lock(this->_request_lock);
        this->_active_requests.push_back(request);
    }
    this->_scheduler->add(request->sequence);
    return request;
}

int64_t Engine::await(const request_t &request) {
    CHECK_ARGUMENT(request != nullptr && request->sequence != nullptr, "invalid request handle");
    std::lock_guard<std::mutex> lock(request->mutex);
    if (!request->sequence->wait_for_token(request->observed_ntoken)) {
        return -2;
    }
    const int64_t token = request->sequence->token_at(request->observed_ntoken);
    ++ request->observed_ntoken;
    return token;
}

void Engine::abort(const request_t &request) {
    if (request && request->sequence) {
        this->_scheduler->abort(request->sequence);
    }
}

void Engine::release(const request_t &request) {
    if (!request) {
        return;
    }
    if (request->sequence && !request->sequence->is_finish()) {
        this->_scheduler->abort(request->sequence);
    }
    std::lock_guard<std::mutex> lock(this->_request_lock);
    auto it = std::find(this->_active_requests.begin(), this->_active_requests.end(), request);
    if (it != this->_active_requests.end()) {
        this->_active_requests.erase(it);
    }
}

};