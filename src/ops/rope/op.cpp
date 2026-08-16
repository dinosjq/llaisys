#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"
#include "nvidia/rope_nvidia.cuh"

namespace llaisys::ops {

namespace {

void rope_check_common(tensor_t out, tensor_t in, tensor_t pos_ids) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(pos_ids->dtype(), LLAISYS_DTYPE_I64);
    ASSERT(out->isContiguous() && in->isContiguous(), "rope: all tensors must be contiguous.");
    ASSERT(out->ndim() == 3, "rope: out must be 3D.");
    ASSERT(in->ndim() == 3, "rope: in must be 3D.");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be 1D.");
    const auto out_shape = out->shape();
    const auto in_shape = in->shape();
    const auto pos_ids_shape = pos_ids->shape();
    CHECK_SAME_SHAPE(out_shape, in_shape);
    ASSERT(out_shape[0] == pos_ids_shape[0], "rope: pos_ids length must match seqlen.");
}

} // namespace

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    rope_check_common(out, in, pos_ids);

    const size_t seqlen = out->shape()[0];
    const size_t head = out->shape()[1];
    const size_t d = out->shape()[2];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seqlen, head, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope(out->data(), in->data(), pos_ids->data(), theta, out->dtype(), seqlen, head, d);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

void rope(tensor_t out, tensor_t in, tensor_t pos_ids, tensor_t inv_freq) {
    rope_check_common(out, in, pos_ids);
    CHECK_SAME_DEVICE(out, inv_freq);
    CHECK_SAME_DTYPE(inv_freq->dtype(), LLAISYS_DTYPE_F32);
    ASSERT(inv_freq->isContiguous(), "rope: inv_freq must be contiguous.");
    ASSERT(inv_freq->ndim() == 1, "rope: inv_freq must be 1D.");
    ASSERT(inv_freq->shape()[0] == out->shape()[2] / 2, "rope: inv_freq length must be dh/2.");

    const size_t seqlen = out->shape()[0];
    const size_t head = out->shape()[1];
    const size_t d = out->shape()[2];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope_inv_freq(out->data(), in->data(), pos_ids->data(), inv_freq->data(), out->dtype(), seqlen,
                                 head, d);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::rope_inv_freq(out->data(), in->data(), pos_ids->data(), inv_freq->data(), out->dtype(), seqlen,
                                     head, d);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
