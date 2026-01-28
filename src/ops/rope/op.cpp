#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, pos_ids);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);
    // TODO: 检查内存连续性 后续貌似要支持不连续的
    ASSERT(out->isContiguous() && in->isContiguous(), "rope: all tensors must be contiguous.");
    // TODO: 检查维度
    ASSERT(out->ndim() == 3, "rope: out must be 3D.");
    ASSERT(in->ndim() == 3, "rope: in must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "rope: weight must be 1D.");
    // TODO: 检查是否符合任务描述
    const auto out_shape = out->shape();
    const auto in_shape = in->shape();
    const auto pos_ids_shape = pos_ids->shape();
    CHECK_SAME_SHAPE(out_shape, in_shape);
    ASSERT(out_shape[0] == pos_ids_shape[0], "rope: pos_ids_shape is not satisfied.");

    const size_t seqlen = out_shape[0];
    const size_t head = out_shape[1];
    const size_t d = out_shape[2];

    // 优先支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的rope
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seqlen, head, d);
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU（冗余，理论上不会走到这里）
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seqlen, head, d);
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
