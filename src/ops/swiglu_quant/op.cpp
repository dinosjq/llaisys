#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "nvidia/swiglu_quant_nvidia.cuh"

namespace llaisys::ops {
void swiglu_quant(tensor_t out_q, tensor_t out_scale, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out_q, out_scale, gate, up);
    ASSERT(out_q->dtype() == LLAISYS_DTYPE_I8, "swiglu_quant: out_q must be I8.");
    ASSERT(out_scale->dtype() == LLAISYS_DTYPE_F32, "swiglu_quant: out_scale must be F32.");
    ASSERT(out_q->isContiguous() && gate->isContiguous() && up->isContiguous(), "swiglu_quant: contiguous required.");
    ASSERT(gate->ndim() == 2 && up->ndim() == 2, "swiglu_quant: gate/up must be 2D.");
    const size_t m = static_cast<size_t>(gate->shape()[0]);
    const size_t k = static_cast<size_t>(gate->shape()[1]);
    ASSERT(static_cast<size_t>(up->shape()[0]) == m && static_cast<size_t>(up->shape()[1]) == k,
           "swiglu_quant: up shape == gate shape.");
    ASSERT(static_cast<size_t>(out_q->shape()[0]) == m && static_cast<size_t>(out_q->shape()[1]) == k,
           "swiglu_quant: out_q shape (m,k).");
    ASSERT(static_cast<size_t>(out_scale->shape()[0]) == m, "swiglu_quant: out_scale length.");
    ASSERT(m > 0 && k > 0, "swiglu_quant: m/k must be > 0.");

    llaisys::core::context().setDevice(gate->deviceType(), gate->deviceId());

    switch (gate->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        EXCEPTION_UNSUPPORTED_DEVICE;
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::swiglu_quant(out_q->data(), out_scale->data(), gate->data(), up->data(), gate->dtype(), m, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
