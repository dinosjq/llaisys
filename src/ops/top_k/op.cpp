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
    // vals 是否连续
    ASSERT(vals->isContiguous() && out_idx->isContiguous() && out_val->isContiguous(), "topk: tensor must be contiguous.");

    std::vector<size_t> shape = vals->shape();
    CHECK_ARGUMENT(shape.size() == 1 || shape.size() == 2, "topk: vals must be a 1D or 2D tensor.");

    const size_t batch_size = shape.size() == 1 ? 1 : shape[0];
    const size_t numel = shape.back();
    CHECK_ARGUMENT(batch_size > 0, "topk: batch size must be positive.");
    CHECK_ARGUMENT(numel > 0, "topk: last dimension must be positive.");
    CHECK_ARGUMENT(k > 0 && k <= numel, "topk: k must be in [1, vals.shape[-1]].");

    std::vector<size_t> out_shape = shape;
    out_shape.back() = k;
    CHECK_ARGUMENT(out_idx->shape() == out_shape, "topk: out_idx shape must match vals except for the last dimension.");
    CHECK_ARGUMENT(out_val->shape() == out_shape, "topk: out_val shape must match vals except for the last dimension.");

    // 设置当前设备
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    // 根据设备类型分发实现
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::topk(out_idx->data(), out_val->data(), vals->data(), vals->dtype(), batch_size, numel, k);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::topk(out_idx->data(), out_val->data(), vals->data(), vals->dtype(), batch_size, numel, k);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
