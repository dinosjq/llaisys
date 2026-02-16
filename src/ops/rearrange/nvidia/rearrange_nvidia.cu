#include "rearrange_nvidia.cuh"

#include "../../../utils.hpp"
#include "../../../utils/nvidia_utils.cuh"

#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

// Kernel implementations
template <int ELEM>
__global__ void rearrange_kernel(uint8_t *out, const uint8_t *in,
                                 const size_t *shape, const ptrdiff_t *out_strides,
                                 const ptrdiff_t *in_strides, const size_t *dim_prod,
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

    // Allocate device copies of small arrays
    size_t *d_shape = nullptr, *d_dim_prod = nullptr;
    ptrdiff_t *d_in_strides = nullptr, *d_out_strides = nullptr;
    cudaError_t cerr = cudaSuccess;
    cerr = cudaMalloc(&d_shape, ndim * sizeof(size_t));
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMalloc d_shape failed");
    cerr = cudaMalloc(&d_dim_prod, ndim * sizeof(size_t));
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMalloc d_dim_prod failed");
    cerr = cudaMalloc(&d_in_strides, ndim * sizeof(ptrdiff_t));
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMalloc d_in_strides failed");
    cerr = cudaMalloc(&d_out_strides, ndim * sizeof(ptrdiff_t));
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMalloc d_out_strides failed");

    cerr = cudaMemcpy(d_shape, shape.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMemcpy d_shape failed");
    cerr = cudaMemcpy(d_dim_prod, dim_prod.data(), ndim * sizeof(size_t), cudaMemcpyHostToDevice);
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMemcpy d_dim_prod failed");
    cerr = cudaMemcpy(d_in_strides, in_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMemcpy d_in_strides failed");
    cerr = cudaMemcpy(d_out_strides, out_strides.data(), ndim * sizeof(ptrdiff_t), cudaMemcpyHostToDevice);
    ASSERT(cerr == cudaSuccess, "rearrange: cudaMemcpy d_out_strides failed");

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
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaFree(d_shape);
    cudaFree(d_dim_prod);
    cudaFree(d_in_strides);
    cudaFree(d_out_strides);
}

} // namespace llaisys::ops::nvidia
