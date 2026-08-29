#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "nvidia/rms_norm_quant_nvidia.cuh"

namespace llaisys::ops {
void rms_norm_quant(tensor_t out_q, tensor_t out_scale, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out_q, out_scale, in, weight);
    ASSERT(out_q->dtype() == LLAISYS_DTYPE_I8, "rms_norm_quant: out_q must be I8.");
    ASSERT(out_scale->dtype() == LLAISYS_DTYPE_F32, "rms_norm_quant: out_scale must be F32.");
    ASSERT(out_q->isContiguous() && in->isContiguous() && weight->isContiguous(),
           "rms_norm_quant: contiguous required.");
    ASSERT(in->ndim() == 2 && weight->ndim() == 1, "rms_norm_quant: in 2D / weight 1D.");
    const size_t m = static_cast<size_t>(in->shape()[0]);
    const size_t k = static_cast<size_t>(in->shape()[1]);
    ASSERT(static_cast<size_t>(out_q->shape()[0]) == m && static_cast<size_t>(out_q->shape()[1]) == k,
           "rms_norm_quant: out_q shape (m,k).");
    ASSERT(static_cast<size_t>(out_scale->shape()[0]) == m, "rms_norm_quant: out_scale length.");
    ASSERT(static_cast<size_t>(weight->shape()[0]) == k, "rms_norm_quant: weight length.");
    ASSERT(m > 0 && k > 0, "rms_norm_quant: m/k must be > 0.");

    llaisys::core::context().setDevice(in->deviceType(), in->deviceId());

    switch (in->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rms_norm_quant(out_q->data(), out_scale->data(), in->data(), weight->data(), in->dtype(), m, k,
                                      eps);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
