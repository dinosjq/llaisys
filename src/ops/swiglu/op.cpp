#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, gate, up);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype(), up->dtype());
    // TODO: 检查内存连续性 后续貌似要支持不连续的
    ASSERT(out->isContiguous() && gate->isContiguous() && up->isContiguous(), "swiglu: all tensors must be contiguous.");
    // TODO: 检查维度
    ASSERT(out->ndim() == 2, "swiglu: out must be 2D.");
    ASSERT(gate->ndim() == 2, "swiglu: gate must be 2D.");
    ASSERT(up->ndim() == 2, "swiglu: up must be 2D.");
    // 检查形状
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());

    // 优先支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的self_attention
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU（冗余，理论上不会走到这里）
        return cpu::swiglu(out->data(), gate->data(), up->data(), out->dtype(), out->numel());
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
