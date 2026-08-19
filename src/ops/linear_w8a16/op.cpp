#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_w8a16_nvidia.cuh"
#endif

namespace llaisys::ops {

void linear_w8a16(tensor_t out, tensor_t in, tensor_t weight_i8, tensor_t scale, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight_i8, scale);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    ASSERT(out->isContiguous() && in->isContiguous() && weight_i8->isContiguous() && scale->isContiguous(),
           "linear_w8a16: contiguous required");
    ASSERT(out->ndim() == 2 && in->ndim() == 2 && weight_i8->ndim() == 2 && scale->ndim() == 1,
           "linear_w8a16: ranks");
    ASSERT(weight_i8->dtype() == LLAISYS_DTYPE_I8, "linear_w8a16: weight I8");
    ASSERT(scale->dtype() == LLAISYS_DTYPE_F32, "linear_w8a16: scale F32");

    const size_t m = out->shape()[0];
    const size_t n = out->shape()[1];
    const size_t k = in->shape()[1];
    ASSERT(in->shape()[0] == m, "linear_w8a16: m");
    ASSERT(weight_i8->shape()[0] == n && weight_i8->shape()[1] == k, "linear_w8a16: weight shape");
    ASSERT(scale->shape()[0] == n, "linear_w8a16: scale");
    if (bias) {
        ASSERT(bias->ndim() == 1 && bias->shape()[0] == n, "linear_w8a16: bias");
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear_w8a16(out->data(), in->data(), weight_i8->data(), scale->data(),
                                    bias ? bias->data() : nullptr, out->dtype(), m, n, k);
#endif
    case LLAISYS_DEVICE_CPU:
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
