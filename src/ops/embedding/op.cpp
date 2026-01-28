#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, index, weight);

    // 判断out的dtype与weight的dtype相同
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_SAME_DTYPE(index->dtype(), LLAISYS_DTYPE_I64);
    
    // 先进行检测连续
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(), "embedding: tensor must be contiguous.");

    // TODO: 似乎还要判断维数
    ASSERT(index->ndim() == 1, "embedding: index must be 1D.");
    ASSERT(weight->ndim() == 2, "embedding: weight must be 2D.");
    ASSERT(out->ndim() == 2, "embedding: out must be 2D.");
    
    // TODO: 似乎要判断index是否超出行数的限制 索引非法
    const int64_t *n_index = reinterpret_cast<const int64_t *>(index->data());
    const size_t idx_numel = index->numel();
    const size_t lim = weight->shape()[0];

    for (size_t i = 0; i < idx_numel; ++ i) {
        int64_t idx = n_index[i];
        ASSERT(idx >= 0 && static_cast<size_t>(idx) < lim, "embedding: idx out of bound.");
    }

    // 判断index的元素数等于out的行数
    ASSERT(idx_numel == out->shape()[0], "embedding: idx_numel != out[0]'s size.");

    // 优先支持CPU计算
    if (weight->deviceType() == LLAISYS_DEVICE_CPU) {
        // 调用CPU实现的embedding
        return cpu::embedding(out->data(), n_index, weight->data(), weight->dtype(), idx_numel, weight->shape()[1], weight->elementSize());
    }

    // 设置当前设备（如GPU等）
    llaisys::core::context().setDevice(weight->deviceType(), weight->deviceId());

    // 根据设备类型分发实现
    switch (weight->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        // 再次处理CPU
        return cpu::embedding(out->data(), n_index, weight->data(), weight->dtype(), idx_numel, weight->shape()[1], weight->elementSize());
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
