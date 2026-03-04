#pragma once
#include "llaisys.h"

#include "../core.hpp"

#include <memory>

namespace llaisys::core {
class Storage {
private:
    std::byte *_memory;
    size_t _size;
    Runtime &_runtime;
    bool _is_host;
    bool _is_first; // 判断是不是内存的起点
    Storage(std::byte *memory, size_t size, Runtime &runtime, bool is_host, bool is_first = true);

public:
    friend class Runtime;
    ~Storage();

    std::byte *memory() const;
    size_t size() const;
    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isHost() const;
};

}; // namespace llaisys::core
