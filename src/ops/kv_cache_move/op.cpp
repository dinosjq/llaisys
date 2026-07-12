#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "nvidia/kv_cache_move_nvidia.cuh"
#include "../../profiler/profiler.hpp"


namespace llaisys::ops {
void kv_cache_move(tensor_t out, tensor_t in, tensor_t block_ids, tensor_t cut_idx, tensor_t pos_ids, size_t max_seq_len)
{
    PROFILE_STEP();
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, block_ids, cut_idx);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(block_ids->dtype() == LLAISYS_DTYPE_I64, "kv_cache_move: block_ids must be i64.");
    // 检查内存连续性
    ASSERT(out->isContiguous() && in->isContiguous() && block_ids->isContiguous(), "kv_cache_move: all tensors must be contiguous.");

    // 检查是否符合任务描述
    const auto out_shape = out->shape();
    const auto block_ids_shape = block_ids->shape();

    const size_t token_num = out_shape[1];
    const size_t nkvh = out_shape[2];
    const size_t dh = out_shape[3];
    const size_t numel = nkvh * dh;
    const size_t batch_size = block_ids_shape[0];
    const size_t max_block_num = block_ids_shape[1];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::kv_cache_move(out->data(), in->data(), block_ids->data(), cut_idx->data(), pos_ids->data(), token_num, batch_size, max_block_num, max_seq_len, out->dtype(), numel);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
