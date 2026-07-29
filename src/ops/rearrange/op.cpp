#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"
#include "nvidia/rearrange_nvidia.cuh"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in);
    // 检查形状一致性
    CHECK_SAME_SHAPE(out->shape(), in->shape());
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());

    if (out->numel() == 0) {
        return;
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rearrange(out->data(), in->data(), out->dtype(), out->shape(), out->strides(), in->strides());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rearrange(out->data(), in->data(), out->dtype(), out->shape(), out->strides(), in->strides());
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
