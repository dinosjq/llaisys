#include "op.hpp"

// 包含核心功能、工具函数和CPU加法实现的头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // 检查数据类型一致
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    // vals 是否连续
    ASSERT(vals->isContiguous() && max_idx->contiguous() && max_val->contiguous(), "argmax: tensor must be contiguous.");

    // 优先支持CPU计算
    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的argmax
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    // 根据设备类型分发实现
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU（冗余，理论上不会走到这里）
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
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
