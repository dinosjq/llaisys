#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "nvidia/linear_w8a8_nvidia.cuh"

namespace llaisys::ops {
void linear_w8a8(tensor_t out, tensor_t in_q, tensor_t a_scale, tensor_t weight_i8, tensor_t w_scale, tensor_t bias)
{
    // 设备一致性
    CHECK_SAME_DEVICE(out, in_q, a_scale, weight_i8, w_scale);
    // 数据类型
    ASSERT(in_q->dtype() == LLAISYS_DTYPE_I8, "linear_w8a8: in_q must be I8.");
    ASSERT(a_scale->dtype() == LLAISYS_DTYPE_F32, "linear_w8a8: a_scale must be F32.");
    ASSERT(weight_i8->dtype() == LLAISYS_DTYPE_I8, "linear_w8a8: weight must be I8.");
    ASSERT(w_scale->dtype() == LLAISYS_DTYPE_F32, "linear_w8a8: w_scale must be F32.");
    // 连续性
    ASSERT(out->isContiguous() && in_q->isContiguous() && weight_i8->isContiguous(), "linear_w8a8: contiguous required.");
    // 形状
    ASSERT(in_q->ndim() == 2 && weight_i8->ndim() == 2, "linear_w8a8: in_q/weight must be 2D.");
    const size_t m = static_cast<size_t>(in_q->shape()[0]);
    const size_t k = static_cast<size_t>(in_q->shape()[1]);
    const size_t n = static_cast<size_t>(out->shape()[1]);
    ASSERT(static_cast<size_t>(weight_i8->shape()[0]) == n && static_cast<size_t>(weight_i8->shape()[1]) == k,
           "linear_w8a8: weight shape (n,k).");
    ASSERT(static_cast<size_t>(a_scale->shape()[0]) == m, "linear_w8a8: a_scale length.");
    ASSERT(static_cast<size_t>(w_scale->shape()[0]) == n, "linear_w8a8: w_scale length.");
    ASSERT(static_cast<size_t>(out->shape()[0]) == m, "linear_w8a8: out m.");
    if (bias) {
        ASSERT(bias->ndim() == 1 && static_cast<size_t>(bias->shape()[0]) == n, "linear_w8a8: bias length.");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear_w8a8(out->data(), in_q->data(), a_scale->data(), weight_i8->data(), w_scale->data(),
                                   bias ? bias->data() : nullptr, out->dtype(), m, n, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
