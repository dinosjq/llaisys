#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../engine/engine.hpp"
#include "../config.hpp"
#include "../tensor/tensor.hpp"
#include "llaisys/runtime.h"
#include "runtime/model_context.hpp"
#include "meta/model_meta.hpp"
#include "weight/model_weights.hpp"
#include "layer/layer.hpp"

namespace llaisys::model {

class Model;
using model_t = std::unique_ptr<Model>;

class Model {
protected:
    // 设备
    llaisysDeviceType_t _device_type = LLAISYS_DEVICE_CPU;
    int _device_id;
    // 元信息
    model_meta_t _meta;
    bool _meta_ready = false;
    // 权重
    model_weights_t _weights;
    bool _weight_ready = false;
    // 上下文（中间张量 / KV cache）
    model_context_t _ctx;
    // 分层 layer
    std::vector<std::unique_ptr<Layer>> _layers;

public:
    virtual ~Model() = default;

    Model(const Model &) = delete;
    Model &operator=(const Model &) = delete;
    Model(Model &&) = delete;
    Model &operator=(Model &&) = delete;

    Model() = default;

    // 前向传播（Worker 调用）
    virtual engine::BatchOutput forward(const engine::BatchInput &input) = 0;

    // load data
    virtual void set_meta(const std::unordered_map<std::string, std::vector<std::uint8_t>> &kv) = 0;
    virtual void set_weight(const std::string &hf_name, tensor_t tensor) = 0;

    // check ready
    bool meta_ready() const { return this->_meta_ready; };
    bool weight_ready() const { return this->_weight_ready; };

    ModelMeta &meta() { return *this->_meta; };
    const ModelMeta &meta() const { return *this->_meta; };

    ModelContext &ctx() { return *this->_ctx; };
    const ModelContext &ctx() const { return *this->_ctx; };

    ModelWeights &weights() { return *this->_weights; };
    const ModelWeights &weights() const { return *this->_weights; };

    // 结束符（Engine 构建 Scheduler 需要）
    virtual int64_t end_token() const = 0;
};

} // namespace llaisys::model
