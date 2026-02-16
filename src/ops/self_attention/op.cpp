#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"
#include "nvidia/self_attention_nvidia.cuh"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    // 检查设备一致性
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    // 检查数据类型一致性
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    // TODO: 检查内存连续性 后续貌似要支持不连续的
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(), "self_attention: all tensors must be contiguous.");
    // TODO: 检查维度
    ASSERT(attn_val->ndim() == 3, "self_attention: attn_val must be 3D.");
    ASSERT(q->ndim() == 3, "self_attention: q must be 3D.");
    ASSERT(k->ndim() == 3, "self_attention: k must be 3D.");
    ASSERT(v->ndim() == 3, "self_attention: v must be 3D.");
    // TODO: 检查是否符合任务描述
    const auto attn_val_shape = attn_val->shape();
    const auto q_shape = q->shape();
    const auto k_shape = k->shape();
    const auto v_shape = v->shape();
    ASSERT(attn_val_shape[0] == q_shape[0], "self_attention: seqlen is not satisfied.");
    ASSERT(attn_val_shape[1] == q_shape[1], "self_attention: nhead is not satisfied.");
    ASSERT(attn_val_shape[2] == v_shape[2], "self_attention: dv is not satisfied.");
    ASSERT(q_shape[2] == k_shape[2], "self_attention: d is not satisfied.");
    ASSERT(k_shape[0] == v_shape[0], "self_attention: total_len is not satisfied.");
    ASSERT(k_shape[1] == v_shape[1], "self_attention: nkvhead is not satisfied.");

    const size_t seqlen = attn_val_shape[0];
    const size_t nhead = attn_val_shape[1];
    const size_t dv = attn_val_shape[2];
    const size_t d = q_shape[2];
    const size_t total_len = k_shape[0];
    const size_t nkvhead = k_shape[1];

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    // 根据设备类型分发实现
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(), 
                                    seqlen, nhead, dv, d, total_len, nkvhead);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), scale, attn_val->dtype(),
                                   seqlen, nhead, dv, d, total_len, nkvhead);
#endif
    default:
        // 不支持的设备类型
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
