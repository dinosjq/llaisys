#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/topk_cpu.hpp"
#include "nvidia/topk_nvidia.cuh"

namespace llaisys::ops {
void topk(tensor_t out_idx, tensor_t out_val, tensor_t vals, size_t k) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out_idx, out_val, vals);
    // 检查数据类型一致
    CHECK_SAME_DTYPE(out_val->dtype(), vals->dtype());
    ASSERT(out_idx->dtype() == LLAISYS_DTYPE_I64 || out_idx->dtype() == LLAISYS_DTYPE_U64,
           "topk: out_idx dtype must be int64 or uint64.");
    ASSERT(k <= vals->numel(), "topk: k is too large.");
    // vals 是否连续
    ASSERT(vals->isContiguous() && out_idx->isContiguous() && out_val->isContiguous(), "topk: tensor must be contiguous.");

    // 设置当前设备
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    // 根据设备类型分发实现
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::topk(out_idx->data(), out_val->data(), vals->data(), vals->dtype(), vals->numel(), k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::topk(out_idx->data(), out_val->data(), vals->data(), vals->dtype(), vals->numel(), k);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
