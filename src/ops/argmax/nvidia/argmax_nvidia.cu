#include "argmax_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <stdexcept>
#include <type_traits>

__device__ unsigned long long warp_reduce(unsigned long long val) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        unsigned long long tmp = __shfl_xor_sync(0xFFFFFFFF, val, offset);
        if (tmp > val) {
            val = tmp;
        }
    }
    return val;
}

/**
 * 保序 解决负数转 uint 因符号位，导致大小顺序反转的问题
 */
__host__ __device__ uint32_t float_to_ordered_uint32(float f) {
    uint32_t u;
#ifdef __CUDA_ARCH__
    u = __float_as_uint(f); 
#else
    u = llaisys::utils::float_to_uint32(f); 
#endif
    const uint32_t sign_mask = 0x80000000;
    if (u & sign_mask) {
        u = ~u;
    } else {
        u |= sign_mask;
    }
    return u;
}

__host__ __device__ float ordered_uint32_to_float(uint32_t u) {
    const uint32_t sign_mask = 0x80000000;
    if (u & sign_mask) {
        u &= ~sign_mask;
    } else {
        u = ~u;
    }
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    return llaisys::utils::uint32_to_float(u);
#endif
}

__device__ unsigned long long nv_merge(float val, size_t idx){
    uint32_t t = float_to_ordered_uint32(val);
    return (static_cast<unsigned long long>(t) << 32) | static_cast<unsigned long long>(idx);
}

static constexpr size_t BLOCK_DIM = 256;
static constexpr size_t SMEM_SIZE = BLOCK_DIM >> 5;

static_assert(BLOCK_DIM % 32 == 0, "BLOCK_DIM illegal");

template <typename T>
__global__ void argmax_kernel(unsigned long long *mx_pack, const T *vals, size_t numel) {
    __shared__ unsigned long long smem[SMEM_SIZE];

    const size_t tid = threadIdx.x;
    const size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + tid;
    const size_t step = static_cast<size_t>(blockDim.x) * gridDim.x;

    unsigned long long mx = nv_merge(-INFINITY, size_t(0));

    for (size_t i = idx; i < numel; i += step) {
        float val = llaisys::utils::nvidia::cast<float>(vals[i]);
        unsigned long long val_idx = nv_merge(val, i);
        if (mx < val_idx) {
            mx = val_idx;
        }
    }

    // warp 归约找最大及最大位置
    mx = warp_reduce(mx);

    // warp 间归约初始化
    if ((tid & 31) == 0) {
        smem[tid >> 5] = mx;
    }

    // block 同步
    __syncthreads();

    // warp 间归约
    if (tid < 32) {
        size_t top = (blockDim.x + 31) >> 5;
        unsigned long long b_mx = tid < top ? smem[tid] : nv_merge(-INFINITY, size_t(0));
        b_mx = warp_reduce(b_mx);

        if (tid == 0) {
            atomicMax(mx_pack, b_mx);
        }
    }
}

/**
 * 将 max_val 与 max_idx 打包在一起，解决 atomicMax 时 max_idx 不会同步修改的问题
 */
template <typename T>
void argmax_launch(std::byte *max_idx, std::byte *max_val, const std::byte *vals, size_t numel) {
    dim3 blockDim(BLOCK_DIM);
    dim3 gridDim(min((numel + blockDim.x - 1) / blockDim.x, size_t(65535)));

    const auto *d_vals = reinterpret_cast<const T *>(vals);
    unsigned long long *d_max_pack = nullptr;    
    unsigned long long init;
    {
        uint32_t v = float_to_ordered_uint32(-INFINITY);
        init = (static_cast<unsigned long long>(v) << 32) | 0u;
    }

    CUDA_CHECK(cudaMalloc(&d_max_pack, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_max_pack, &init, sizeof(init), cudaMemcpyHostToDevice));

    argmax_kernel<<<gridDim, blockDim>>>(d_max_pack, d_vals, numel);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long result;
    CUDA_CHECK(cudaMemcpy(&result, d_max_pack, sizeof(result), cudaMemcpyDeviceToHost));

    uint32_t t_val = static_cast<uint32_t>(result >> 32);
    uint32_t t_idx = static_cast<uint32_t>(result & 0xFFFFFFFFu);
    T val = llaisys::utils::cast<T>(ordered_uint32_to_float(t_val));
    size_t idx = static_cast<size_t>(t_idx);

    CUDA_CHECK(cudaMemcpy(max_val, &val, sizeof(val), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(max_idx, &idx, sizeof(idx), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaFree(d_max_pack));
}

namespace llaisys::ops::nvidia {
void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t numel) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_launch<float>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_BF16:
        return argmax_launch<llaisys::bf16_t>(max_idx, max_val, vals, numel);
    case LLAISYS_DTYPE_F16:
        return argmax_launch<llaisys::fp16_t>(max_idx, max_val, vals, numel);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::nvidia
