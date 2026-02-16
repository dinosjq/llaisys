#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"
#include "nvidia/linear_nvidia.cuh"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, weight);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    // 检查内存连续性
    ASSERT(out->isContiguous() && in->isContiguous() && weight->isContiguous(), "linear: all tensors must be contiguous.");
    // TODO: 检查维度
    ASSERT(out->ndim() == 2, "linear: out must be 2D.");
    ASSERT(in->ndim() == 2, "linear: in must be 2D.");
    ASSERT(weight->ndim() == 2, "linear: weight must be 2D.");
    // TODO: 检查是否符合矩阵乘法
    const auto &out_shape = out->shape();
    const auto &in_shape = in->shape();
    const auto &weight_shape = weight->shape();
    ASSERT(out_shape[0] == in_shape[0], "linear: shape error 0.");
    ASSERT(out_shape[1] == weight_shape[0], "linear: shape error 1.");
    ASSERT(in_shape[1] == weight_shape[1], "linear: shape error 2.");

    const size_t n = out_shape[0];
    const size_t m = out_shape[1];
    const size_t t = in_shape[1];

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    const std::byte *d_bias = nullptr;
    if(bias != nullptr)
        d_bias = bias->data();

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), n, m, t);
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), out->dtype(), n, m, t);
        }
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(), d_bias, out->dtype(), n, m, t);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
