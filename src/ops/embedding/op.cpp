#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"
#include "nvidia/embedding_nvidia.cuh"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, index, weight);

    // 判断out的dtype与weight的dtype相同
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);
    
    // 先进行检测连续
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "embedding: tensor must be contiguous.");

    // 还要判断维数
    ASSERT(index->ndim() == 1, "embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D.");
    
    // 设置当前设备
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    // 根据设备类型分发实现
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1], weight->elementSize());
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::embedding(out->data(), index->data(), weight->data(), index->numel(), weight->shape()[1], weight->elementSize());
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
