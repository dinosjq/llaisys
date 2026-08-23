#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/dequant_w8_nvidia.cuh"
#endif

namespace llaisys::ops {

void dequant_w8(tensor_t out, tensor_t weight_i8, tensor_t scale) {
    CHECK_SAME_DEVICE(out, weight_i8, scale);
    ASSERT(out->isContiguous() && weight_i8->isContiguous() && scale->isContiguous(),
           "dequant_w8: tensors must be contiguous");
    ASSERT(out->ndim() == 2 && weight_i8->ndim() == 2 && scale->ndim() == 1, "dequant_w8: shape rank");
    ASSERT(weight_i8->dtype() == LLAISYS_DTYPE_I8, "dequant_w8: weight must be I8");
    ASSERT(scale->dtype() == LLAISYS_DTYPE_F32, "dequant_w8: scale must be F32");
    ASSERT(out->shape()[0] == weight_i8->shape()[0] && out->shape()[1] == weight_i8->shape()[1],
           "dequant_w8: out/weight shape");
    ASSERT(scale->shape()[0] == weight_i8->shape()[0], "dequant_w8: scale length");

    const size_t n = weight_i8->shape()[0];
    const size_t k = weight_i8->shape()[1];
    (void)n;
    (void)k;
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::dequant_w8(out->data(), weight_i8->data(), scale->data(), out->dtype(), n, k);
#endif
    case LLAISYS_DEVICE_CPU:
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
