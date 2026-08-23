#pragma once

#include "../../model/model.hpp"
#include "../engine.hpp"
#include "../../sequence/sequence.hpp"

namespace llaisys::engine{

class Worker{
private:
    // 设备
    llaisysDeviceType_t _device_type = LLAISYS_DEVICE_CPU;
    int _device_id;
    // 模型
    model::model_t _model;
    // 输入、输出
    std::queue<BatchInput> _tasks;
    std::queue<BatchOutput> _results;
    // 执行
    bool _running = false;
    std::thread _runner;

    void _start();
    void _stop();

public:
    Worker(llaisysDeviceType_t device_type, int device_id, model::model_t model);
    ~Worker() = default;

    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;
    Worker(Worker &&) = delete;
    Worker &operator=(Worker &&) = delete;

    // 预处理
    BatchInput prepare(const std::vector<seq_t> &seqs, const size_t table_width, const bool is_prefill);
    // 前向
    BatchOutput forward(const BatchInput &input);
    
    // 模型
    bool is_ready() { return this->_model->meta_ready() && this->_model->weight_ready(); };
    model::Model &model() { return *this->_model; };
    const model::Model &model() const { return *this->_model; };
};

};