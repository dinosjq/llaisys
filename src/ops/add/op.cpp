
// 加法算子实现文件
#include "op.hpp"

// 包含核心功能、工具函数和CPU加法实现的头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/add_cpu.hpp"

namespace llaisys::ops {

//
// add: 实现张量加法操作 c = a + b
//
void add(tensor_t c, tensor_t a, tensor_t b) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(c, a, b);
    // 检查形状一致性
    CHECK_SAME_SHAPE(c->shape(), a->shape(), b->shape());
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(c->dtype(), a->dtype(), b->dtype());
    // 检查内存连续性
    ASSERT(c->isContiguous() && a->isContiguous() && b->isContiguous(), "Add: all tensors must be contiguous.");

    // 优先支持CPU计算
    if (c->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的加法
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(c->deviceType(), c->deviceId());

    // 根据设备类型分发实现
    switch (c->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU（冗余，理论上不会走到这里）
        return cpu::add(c->data(), a->data(), b->data(), c->dtype(), c->numel());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        // NVIDIA GPU实现待补充
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
