#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "../../utils/types.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/quantize_w8_nvidia.cuh"
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace llaisys::ops {
namespace {

void quantize_w8_cpu(tensor_t weight_i8, tensor_t scale, tensor_t weight_fp) {
    const size_t n = weight_fp->shape()[0];
    const size_t k = weight_fp->shape()[1];
    const auto dtype = weight_fp->dtype();
    const std::byte *base = weight_fp->data();
    const size_t esize = utils::dsize(dtype);

    auto load_f32 = [&](size_t idx) -> float {
        const std::byte *p = base + idx * esize;
        switch (dtype) {
        case LLAISYS_DTYPE_F32:
            return *reinterpret_cast<const float *>(p);
        case LLAISYS_DTYPE_BF16:
            return utils::cast<float>(*reinterpret_cast<const bf16_t *>(p));
        case LLAISYS_DTYPE_F16:
            return utils::cast<float>(*reinterpret_cast<const fp16_t *>(p));
        default:
            throw std::runtime_error("quantize_w8 cpu: unsupported dtype");
        }
    };

    auto *q = reinterpret_cast<int8_t *>(weight_i8->data());
    auto *s = reinterpret_cast<float *>(scale->data());
    for (size_t i = 0; i < n; ++i) {
        float amax = 0.f;
        for (size_t j = 0; j < k; ++j) {
            amax = std::max(amax, std::fabs(load_f32(i * k + j)));
        }
        const float sc = (amax == 0.f) ? 1.f : (amax / 127.f);
        s[i] = sc;
        for (size_t j = 0; j < k; ++j) {
            const float v = std::nearbyintf(load_f32(i * k + j) / sc);
            const int clipped = std::max(-128, std::min(127, static_cast<int>(v)));
            q[i * k + j] = static_cast<int8_t>(clipped);
        }
    }
}

} // namespace

void quantize_w8(tensor_t weight_i8, tensor_t scale, tensor_t weight_fp) {
    CHECK_SAME_DEVICE(weight_i8, scale, weight_fp);
    ASSERT(weight_i8->isContiguous() && scale->isContiguous() && weight_fp->isContiguous(),
           "quantize_w8: contiguous required");
    ASSERT(weight_fp->ndim() == 2 && weight_i8->ndim() == 2 && scale->ndim() == 1, "quantize_w8: ranks");
    ASSERT(weight_i8->dtype() == LLAISYS_DTYPE_I8, "quantize_w8: out I8");
    ASSERT(scale->dtype() == LLAISYS_DTYPE_F32, "quantize_w8: scale F32");
    ASSERT(weight_i8->shape()[0] == weight_fp->shape()[0] && weight_i8->shape()[1] == weight_fp->shape()[1],
           "quantize_w8: weight shape");
    ASSERT(scale->shape()[0] == weight_fp->shape()[0], "quantize_w8: scale length");

    const size_t n = weight_fp->shape()[0];
    const size_t k = weight_fp->shape()[1];
    llaisys::core::context().setDevice(weight_fp->deviceType(), weight_fp->deviceId());

    switch (weight_fp->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return quantize_w8_cpu(weight_i8, scale, weight_fp);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::quantize_w8(weight_i8->data(), scale->data(), weight_fp->data(), weight_fp->dtype(), n, k);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}

} // namespace llaisys::ops
