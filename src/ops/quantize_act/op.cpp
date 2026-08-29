#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "nvidia/quantize_act_nvidia.cuh"

namespace llaisys::ops {
void quantize_act(tensor_t out_q, tensor_t out_scale, tensor_t in)
{
    // 设备一致性
    CHECK_SAME_DEVICE(out_q, out_scale, in);
    // 数据类型
    ASSERT(out_q->dtype() == LLAISYS_DTYPE_I8, "quantize_act: out_q must be I8.");
    ASSERT(out_scale->dtype() == LLAISYS_DTYPE_F32, "quantize_act: out_scale must be F32.");
    // 连续性
    ASSERT(out_q->isContiguous() && in->isContiguous(), "quantize_act: contiguous required.");
    // 形状
    ASSERT(in->ndim() == 2, "quantize_act: in must be 2D.");
    const size_t m = static_cast<size_t>(in->shape()[0]);
    const size_t k = static_cast<size_t>(in->shape()[1]);
    ASSERT(static_cast<size_t>(out_q->shape()[0]) == m && static_cast<size_t>(out_q->shape()[1]) == k,
           "quantize_act: out_q shape (m,k).");
    ASSERT(static_cast<size_t>(out_scale->shape()[0]) == m, "quantize_act: out_scale length.");
    ASSERT(k > 0 && m > 0, "quantize_act: m/k must be > 0.");

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::quantize_act(out_q->data(), out_scale->data(), in->data(), in->dtype(), m, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
