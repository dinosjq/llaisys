#include "rearrange_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/nvidia_utils.cuh"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// Kernel implementations
template <int ELEM>
__global__ void rearrange_kernel(uint8_t *__restrict__ out, const uint8_t *__restrict__ in,
                                 const size_t *__restrict__ shape, const ptrdiff_t *__restrict__ out_strides,
                                 const ptrdiff_t *__restrict__ in_strides, const size_t *__restrict__ dim_prod,
                                 int ndim, size_t numel) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        size_t tmp = idx;
        ptrdiff_t in_off = 0;
        ptrdiff_t out_off = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t coord = tmp / dim_prod[d];
            tmp = tmp % dim_prod[d];
            in_off += static_cast<ptrdiff_t>(coord) * in_strides[d];
            out_off += static_cast<ptrdiff_t>(coord) * out_strides[d];
        }
        uint8_t *dst = out + out_off * ELEM;
        const uint8_t *src = in + in_off * ELEM;
        if constexpr (ELEM == 1) {
            dst[0] = src[0];
        } else if constexpr (ELEM == 2) {
            *reinterpret_cast<uint16_t *>(dst) = *reinterpret_cast<const uint16_t *>(src);
        } else if constexpr (ELEM == 4) {
            *reinterpret_cast<uint32_t *>(dst) = *reinterpret_cast<const uint32_t *>(src);
        } else if constexpr (ELEM == 8) {
            *reinterpret_cast<uint64_t *>(dst) = *reinterpret_cast<const uint64_t *>(src);
        }
    }
}

__global__ void rearrange_kernel_generic(uint8_t *out, const uint8_t *in,
                                         const size_t *shape, const ptrdiff_t *out_strides,
                                         const ptrdiff_t *in_strides, const size_t *dim_prod,
                                         int ndim, size_t numel, int elem_size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = gridDim.x * blockDim.x;
    for (; idx < numel; idx += stride) {
        size_t tmp = idx;
        ptrdiff_t in_off = 0;
        ptrdiff_t out_off = 0;
        for (int d = 0; d < ndim; ++d) {
            size_t coord = tmp / dim_prod[d];
            tmp = tmp % dim_prod[d];
            in_off += static_cast<ptrdiff_t>(coord) * in_strides[d];
            out_off += static_cast<ptrdiff_t>(coord) * out_strides[d];
        }
        uint8_t *dst = out + out_off * elem_size;
        const uint8_t *src = in + in_off * elem_size;
        for (int b = 0; b < elem_size; ++b) {
            dst[b] = src[b];
        }
    }
}

void rearrange(std::byte *out, const std::byte *in, llaisysDataType_t dtype,
               const std::vector<size_t> &shape,
               const std::vector<ptrdiff_t> &out_strides,
               const std::vector<ptrdiff_t> &in_strides) {
    const size_t ndim = shape.size();
    if (ndim == 0) {
        return;
    }
    ASSERT(out_strides.size() == ndim && in_strides.size() == ndim, "rearrange: strides size mismatch.");

    size_t numel = 1;
    for (size_t d = 0; d < ndim; ++d) {
        numel *= shape[d];
    }
    if (numel == 0) {
        return;
    }

    const size_t elem_size = llaisys::utils::dsize(dtype);

    // Precompute dim products: prod[d] = product_{i=d+1..ndim-1} shape[i]
    std::vector<size_t> dim_prod(ndim);
    for (size_t d = 0; d < ndim; ++d) {
        size_t p = 1;
        for (size_t k = d + 1; k < ndim; ++k) {
            p *= shape[k];
        }
        dim_prod[d] = p;
    }

    // Use thread_local pre-allocated device buffers to avoid cudaMalloc/cudaFree on every call.
    // Each buffer holds at most MAX_NDIM elements (more than enough for any tensor).
    static constexpr size_t MAX_NDIM = 16;
    static thread_local size_t *d_shape = nullptr;
    static thread_local size_t *d_dim_prod = nullptr;
    static thread_local ptrdiff_t *d_in_strides = nullptr;
    static thread_local ptrdiff_t *d_out_strides = nullptr;
    static thread_local bool bufs_inited = false;
    if (!bufs_inited) {
        CUDA_CHECK(cudaMalloc(&d_shape, MAX_NDIM * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_dim_prod, MAX_NDIM * sizeof(size_t)));
        CUDA_CHECK(cudaMalloc(&d_in_strides, MAX_NDIM * sizeof(ptrdiff_t)));
        CUDA_CHECK(cudaMalloc(&d_out_strides, MAX_NDIM * sizeof(ptrdiff_t)));
        bufs_inited = true;
    }

    ASSERT(ndim <= MAX_NDIM, "rearrange: ndim exceeds MAX_NDIM");

    // Fast path: if in/out have same strides (simple contiguous copy), use cudaMemcpyAsync.
    bool same_strides = true;
    for (size_t d = 0; d < ndim; ++d) {
        if (in_strides[d] != out_strides[d]) { same_strides = false; break; }
    }
    if (same_strides) {
        CUDA_CHECK(cudaMemcpyAsync(out, in, numel * elem_size, cudaMemcpyDeviceToDevice, 0));
        CUDA_CHECK(cudaGetLastError());
        return;
    }

    CUDA_CHECK(cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dim_prod, dim_prod.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_in_strides, in_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice));

    const int block = 256;
    const int grid = static_cast<int>((numel + block - 1) / block);

    auto launch_kernel = [&](int esize) {
        switch (esize) {
        case 1:
            rearrange_kernel<1><<<grid, block>>>(
                reinterpret_cast<uint8_t *>(out), reinterpret_cast<const uint8_t *>(in),
                d_shape, d_out_strides, d_in_strides, d_dim_prod, static_cast<int>(ndim), numel);
            break;
        case 2:
            rearrange_kernel<2><<<grid, block>>>(
                reinterpret_cast<uint8_t *>(out), reinterpret_cast<const uint8_t *>(in),
                d_shape, d_out_strides, d_in_strides, d_dim_prod, static_cast<int>(ndim), numel);
            break;
        case 4:
            rearrange_kernel<4><<<grid, block>>>(
                reinterpret_cast<uint8_t *>(out), reinterpret_cast<const uint8_t *>(in),
                d_shape, d_out_strides, d_in_strides, d_dim_prod, static_cast<int>(ndim), numel);
            break;
        case 8:
            rearrange_kernel<8><<<grid, block>>>(
                reinterpret_cast<uint8_t *>(out), reinterpret_cast<const uint8_t *>(in),
                d_shape, d_out_strides, d_in_strides, d_dim_prod, static_cast<int>(ndim), numel);
            break;
        default:
            rearrange_kernel_generic<<<grid, block>>>(
                reinterpret_cast<uint8_t *>(out), reinterpret_cast<const uint8_t *>(in),
                d_shape, d_out_strides, d_in_strides, d_dim_prod, static_cast<int>(ndim), numel, esize);
        }
    };

    launch_kernel(static_cast<int>(elem_size));

    CUDA_CHECK(cudaGetLastError());
}

} // namespace llaisys::ops::nvidia
