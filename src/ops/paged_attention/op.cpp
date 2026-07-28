#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../profiler/profiler.hpp"
#include "../../utils.hpp"
#include "../../config.hpp"

#include "cpu/paged_attention_cpu.hpp"
#include "nvidia/paged_attention_nvidia.cuh"
#include "nvidia/flash_decoding_nvidia.cuh"

namespace llaisys::ops {
void paged_attention(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache, tensor_t block_ids, tensor_t cut_idx, tensor_t tot_len, size_t max_seq_len, float scale, bool is_prefill)
{
    PROFILE_STEP();
    // 检查设备一致性
    CHECK_SAME_DEVICE(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k_cache->dtype(), v_cache->dtype());
    ASSERT(block_ids->dtype() == LLAISYS_DTYPE_I64, "paged_attention: block_ids must be i64.");
    // 检查内存连续性
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k_cache->isContiguous() && v_cache->isContiguous() && block_ids->isContiguous(), "paged_attention: all tensors must be contiguous.");
    // 检查维度
    ASSERT(attn_val->ndim() == 3, "paged_attention: attn_val must be 3D.");
    ASSERT(q->ndim() == 3, "paged_attention: q must be 3D.");
    ASSERT(k_cache->ndim() == 4, "paged_attention: k_cache must be 4D.");
    ASSERT(v_cache->ndim() == 4, "paged_attention: v_cache must be 4D.");
    ASSERT(block_ids->ndim() == 2, "paged_attention: block_ids must be 2D (batch, max_block_num).");
    // 检查是否符合任务描述
    const auto attn_val_shape = attn_val->shape();
    const auto q_shape = q->shape();
    const auto k_shape = k_cache->shape();
    const auto v_shape = v_cache->shape();
    const auto block_ids_shape = block_ids->shape();

    ASSERT(attn_val_shape[0] == q_shape[0], "paged_attention: tot_seqlen is not satisfied.");
    ASSERT(attn_val_shape[1] == q_shape[1], "paged_attention: nh is not satisfied.");
    ASSERT(attn_val_shape[2] == v_shape[3], "paged_attention: dv is not satisfied.");
    ASSERT(q_shape[2] == k_shape[3], "paged_attention: d is not satisfied.");
    ASSERT(k_shape[2] == v_shape[2], "paged_attention: nkvh is not satisfied.");
    ASSERT(k_shape[0] == v_shape[0], "paged_attention: block_num is not satisfied.");
    ASSERT(k_shape[1] == v_shape[1], "paged_attention: token_num is not satisfied.");
    ASSERT(k_shape[2] != 0 && q_shape[1] % k_shape[2] == 0, "paged_attention: nh and nkvh are not satisfied.");

    const size_t nh = attn_val_shape[1];
    const size_t dv = attn_val_shape[2];
    const size_t d = q_shape[2];
    const size_t token_num = k_shape[1];
    const size_t nkvh = k_shape[2];
    
    const size_t batch_size = block_ids_shape[0];
    const size_t max_block_num = block_ids_shape[1];

#ifdef ENABLE_NVIDIA_API
    static llaisysDeviceType_t device = attn_val->deviceType();
    static int device_id = attn_val->deviceId();
    static tensor_t attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dv}, LLAISYS_DTYPE_F32, device, device_id);
    static tensor_t attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
    static tensor_t attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
#endif

    ASSERT(cut_idx->ndim() == 1 && cut_idx->shape()[0] == batch_size + 1, "paged_attention: cut_idx must be 1D with length batch+1.");
    ASSERT(tot_len->ndim() == 1 && tot_len->shape()[0] == batch_size, "paged_attention: tot_len must be 1D with length batch.");

    // 根据设备类型分发实现
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::paged_attention(attn_val->data(), q->data(), k_cache->data(), v_cache->data(), block_ids->data(), cut_idx->data(), tot_len->data(),
                                        token_num, batch_size, max_block_num, max_seq_len, attn_val->dtype(), scale, nh, dv, d, nkvh);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        if(is_prefill){
            return nvidia::paged_attention(attn_val->data(), q->data(), k_cache->data(), v_cache->data(), block_ids->data(), cut_idx->data(), tot_len->data(),
                                    token_num, batch_size, max_block_num, max_seq_len, attn_val->dtype(), scale, nh, dv, d, nkvh);
        } else {
            return nvidia::flash_decoding(attn_val->data(), attn_acc->data(), attn_sum->data(), attn_max->data(),  q->data(), k_cache->data(), 
                                        v_cache->data(), block_ids->data(), cut_idx->data(), tot_len->data(), token_num, batch_size, max_block_num, 
                                        max_seq_len, attn_val->dtype(), scale, nh, dv, d, nkvh);
        }
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
