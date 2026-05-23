#pragma once
#include "../../include/llaisys/models/qwen2.h"
#include "../tensor/tensor.hpp"
#include "../kv_cache/block_manager.hpp"

namespace llaisys{

class Qwen2;
using Qwen2_t = std::shared_ptr<Qwen2>;

struct Qwen2Meta {
    llaisysDataType_t dtype;
    size_t nlayer, hs, nh, nkvh, dh, di, maxseq, voc;
    float epsilon, theta;
    int64_t end_token;
};

using Qwen2Weights = ::LlaisysQwen2Weights;

class Qwen2{
private:
    Qwen2Meta _meta;
    llaisysDeviceType_t _device_type;
    std::vector<int> _device_ids;
    Qwen2Weights _weights;
    BlockManager_t _manager;

    Qwen2(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

public:
    static Qwen2_t model(Qwen2Meta meta, llaisysDeviceType_t device, int *device_ids, int ndevice);

    ~Qwen2();

    // Prevent copy
    Qwen2(const Qwen2 &) = delete;
    Qwen2 &operator=(const Qwen2 &) = delete;

    // Prevent move
    Qwen2(Qwen2 &&) = delete;
    Qwen2 &operator=(Qwen2 &&) = delete;

    // 前向传播
    int64_t forward(const int64_t *token_ids, const size_t ntoken);

    // 元信息
    Qwen2Meta meta(){ return this->_meta; }

    // 模型权重
    Qwen2Weights &weights(){ return this->_weights; }
    const Qwen2Weights &weights() const { return this->_weights; }

    // 设备信息
    llaisysDeviceType_t device_type(){ return this->_device_type; }
    int device_id(){ return this->_device_ids[0]; }

    // kv cache 管理
    BlockManager_t &KVcacheManager(){ return this->_manager; }
};

};