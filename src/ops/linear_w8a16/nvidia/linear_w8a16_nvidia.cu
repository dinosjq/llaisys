#include "linear_w8a16_nvidia.cuh"

#include "gemv_w8a16_nvidia.cuh"
#include "../../dequant_w8/nvidia/dequant_w8_nvidia.cuh"
#include "../../linear/nvidia/linear_nvidia.cuh"
#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <cstdint>
#include <mutex>
#include <stdexcept>

namespace {

size_t dtype_nbytes(llaisysDataType_t dtype) {
    switch (dtype) {
    case LLAISYS_DTYPE_F32:
        return 4;
    case LLAISYS_DTYPE_F16:
    case LLAISYS_DTYPE_BF16:
        return 2;
    default:
        throw std::runtime_error("linear_w8a16: unsupported dtype for workspace");
    }
}

std::byte *dequant_workspace(size_t bytes) {
    static std::mutex mu;
    static std::byte *ptr = nullptr;
    static size_t capacity = 0;
    std::lock_guard<std::mutex> lock(mu);
    if (bytes > capacity) {
        if (ptr != nullptr) {
            CUDA_CHECK(cudaFree(ptr));
            ptr = nullptr;
            capacity = 0;
        }
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&ptr), bytes));
        capacity = bytes;
    }
    return ptr;
}

} // namespace

namespace llaisys::ops::nvidia {

void linear_w8a16(std::byte *out, const std::byte *in, const std::byte *weight_i8, const std::byte *scale,
                  const std::byte *bias, llaisysDataType_t dtype, size_t m, size_t n, size_t k) {
    if (gemv_w8a16_supported(dtype, m, n, k)) {
        return gemv_w8a16(out, in, weight_i8, scale, bias, dtype, m, n, k);
    }

    const size_t bytes = n * k * dtype_nbytes(dtype);
    std::byte *fp_weight = dequant_workspace(bytes);
    dequant_w8(fp_weight, weight_i8, scale, dtype, n, k);
    linear(out, in, fp_weight, bias, dtype, m, n, k);
}

} // namespace llaisys::ops::nvidia
