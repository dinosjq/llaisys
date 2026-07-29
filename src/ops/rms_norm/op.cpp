#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"
#include "nvidia/rms_norm_nvidia.cuh"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, weight);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 检查内存连续性
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "rms_norm: all tensors must be contiguous.");
    // TODO: 检查维度
    ASSERT(out->ndim() == 2, "rms_norm: out must be 2D.");
    ASSERT(in->ndim() == 2, "rms_norm: in must be 2D.");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be 1D.");
    // TODO: 检查是否符合任务描述
    const auto out_shape = out->shape();
    const auto in_shape = in->shape();
    const auto weight_shape = weight->shape();
    ASSERT(out_shape == in_shape, "rms_norm: out_shape != in_shape.");
    ASSERT(in_shape[1] == weight_shape[0], "rms_norm: weight_shape is not satisfied.");

    const size_t n = out_shape[0];
    const size_t m = out_shape[1];

    // 设置当前设备
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), n, m);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm(out->data(), in->data(), weight->data(), eps, out->dtype(), n, m);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
