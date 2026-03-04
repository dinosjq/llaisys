#pragma once
#include "../core.hpp"

#include "../../device/runtime_api.hpp"
#include "../allocator/allocator.hpp"

#include<vector>

namespace llaisys::core {
class Runtime {
private:
    llaisysDeviceType_t _device_type;
    int _device_id;
    const LlaisysRuntimeAPI *_api;
    MemoryAllocator *_allocator;
    bool _is_active;
    void _activate();
    void _deactivate();
    llaisysStream_t _stream;
    Runtime(llaisysDeviceType_t device_type, int device_id);

public:
    friend class Context;

    ~Runtime();

    // Prevent copying
    Runtime(const Runtime &) = delete;
    Runtime &operator=(const Runtime &) = delete;

    // Prevent moving
    Runtime(Runtime &&) = delete;
    Runtime &operator=(Runtime &&) = delete;

    llaisysDeviceType_t deviceType() const;
    int deviceId() const;
    bool isActive() const;

    const LlaisysRuntimeAPI *api() const;

    storage_t allocateDeviceStorage(size_t size);
    storage_t allocateHostStorage(size_t size);
    /**
     * 参考内存池的思路：统一分配内存防止碎片和多次释放的开销
     * TODO: kv cache 专用
     */
    std::vector<storage_t> allocateKVStorage(size_t nlayer, size_t nsize);

    /**
     * TODO: 多层注意力中间矩阵专用
     */
    std::vector<storage_t> allocateMPStorage(const std::vector<size_t> &sizes);

    void freeStorage(Storage *storage);

    llaisysStream_t stream() const;
    void synchronize() const;

};
} // namespace llaisys::core
