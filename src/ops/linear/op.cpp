// 矩阵乘法算子实现文件
#include "op.hpp"

// 包含核心功能、工具函数和CPU矩阵乘法实现的头文件
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

// 傻逼了，忘了判bias为null了

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
    // bias 单独判断
    if(bias != nullptr){
        // std::cout << "存在" << std::endl;
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
        ASSERT(bias->isContiguous(), "linear: bias must be contiguous.");
        ASSERT(bias->ndim() == 1, "linear: bias must be 1D.");
        const auto &bias_shape = bias->shape();
        ASSERT(out_shape[1] == bias_shape[0], "linear: shape error 2.");
    }
    const size_t n = out_shape[0];
    const size_t m = out_shape[1];
    const size_t t = in_shape[1];

    // 优先支持CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的矩阵乘法
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), n, m, t);
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), out->dtype(), n, m, t);
        }
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU（冗余，理论上不会走到这里）
        if (bias != nullptr) {
            return cpu::linear(out->data(), in->data(), weight->data(), bias->data(), out->dtype(), n, m, t);
        } else {
            return cpu::linear(out->data(), in->data(), weight->data(), out->dtype(), n, m, t);
        }
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
