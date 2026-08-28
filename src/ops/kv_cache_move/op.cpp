#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../config.hpp"

#include "nvidia/kv_cache_move_nvidia.cuh"


namespace llaisys::ops {
void kv_cache_move(tensor_t out, tensor_t in, tensor_t block_ids, tensor_t cut_idx, tensor_t pos_ids)
{
    // 检查设备一致性
    CHECK_SAME_DEVICE(out, in, block_ids, cut_idx);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(block_ids->dtype() == LLAISYS_DTYPE_I64, "kv_cache_move: block_ids must be i64.");
    // 检查内存连续性
    ASSERT(out->isContiguous() && in->isContiguous() && block_ids->isContiguous(), "kv_cache_move: all tensors must be contiguous.");

    // 检查是否符合任务描述
    const auto out_shape = out->shape();
    const auto in_shape = in->shape();
    const auto block_ids_shape = block_ids->shape();

    (void)out_shape; (void)in_shape; (void)block_ids_shape;   // 形状仅用于校验（TODO: 维度检查）
    const size_t block_size = out_shape[1];   // 每块 token 数
    const size_t nkvh = out_shape[2];
    const size_t dh = out_shape[3];
    const size_t numel = nkvh * dh;
    const size_t tot_token_num = in_shape[0];   // 本批次待搬移 token 总数
    const size_t max_block_num = block_ids_shape[1];
    ASSERT(block_size > 0, "kv_cache_move: block_size must be > 0.");
    ASSERT(tot_token_num > 0, "kv_cache_move: tot_token_num must be > 0.");
    ASSERT(max_block_num > 0, "kv_cache_move: max_block_num must be > 0.");
    ASSERT(numel > 0, "kv_cache_move: numel must be > 0.");

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    // 根据设备类型分发实现
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::kv_cache_move(out->data(), in->data(), block_ids->data(), cut_idx->data(), pos_ids->data(),
                                 block_size, tot_token_num, max_block_num, out->dtype(), numel);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
