#include "linear_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace {
/**
 * TODO:
 * 调用 cublas 的 api 实现 linear
 * 还有自己实现 linear 且做性能对比
 */

// blockDim
static constexpr size_t BLOCK_SIZE = 256;
static constexpr size_t BK_M = 16;
static constexpr size_t BK_N = 16;

// smem size
static constexpr size_t SMEM_SIZE = 1024;
static constexpr size_t LOAD_COUNT = SMEM_SIZE / BLOCK_SIZE;

// thread
static constexpr size_t TM = 8;
static constexpr size_t TN = 8;

// block tiling
static constexpr size_t BM = TM * BK_M;
static constexpr size_t BN = TN * BK_N;
static constexpr size_t BK = SMEM_SIZE / BM;

/**
 * DIM: C: [ M x N ], A: [ M x K ], B: [ N x K ], BIAS: [ N ]
 * GEMM: 2D-block tiling, shared memory, thread tiling, 向量化访存 ...
 *
 * SMEM LOAD 不用向量化访存原因:
 *   向量化访存，要求大小为4的倍数，保证通用性，所以load到SMEM不采用向量化访存
 *   并且模板类型无法直接转化为float4类型，根本原因是没法用，back to C同理
 * 
 * TODO: 解决 bank Conflicts z-order, 双缓冲 ...
 * 
 * 最后写法优化 彻底丧失可读性
 */

#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])

template <typename T>
__global__ void linear_kernel(T *C, const T *A, const T *B, const T *bias,
                              const size_t M, const size_t N, const size_t K)
{
    // smem
    __shared__ float SM_A[BK][BM];
    __shared__ float SM_B[BK][BN];

    // block
    const size_t BM_OFFSET = blockIdx.y * BM;
    const size_t BN_OFFSET = blockIdx.x * BN;

    // thread
    const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
    const size_t TM_OFFSET = threadIdx.y * TM;
    const size_t TN_OFFSET = threadIdx.x * TN;

    // compute result
    float result[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            result[i][j] = float(0);
        }

    for (size_t BK_OFFSET = 0; BK_OFFSET < K; BK_OFFSET += BK) {
        // load A and B tile
#pragma unroll
        for (size_t i = 0; i < LOAD_COUNT; ++ i) {
            size_t idx = tid + BLOCK_SIZE * i;

            const size_t SA_ROW = idx / BK;
            const size_t SA_COL = idx % BK;

            const size_t A_ROW = BM_OFFSET + SA_ROW;
            const size_t A_COL = BK_OFFSET + SA_COL;

            const size_t A_IDX = A_ROW * K + A_COL;
            SM_A[SA_COL][SA_ROW] = (A_ROW < M && A_COL < K) ? llaisys::utils::nvidia::cast<float>(A[A_IDX]) : float(0);
        }

#pragma unroll
        for (size_t i = 0; i < LOAD_COUNT; ++i) {
            size_t idx = tid + BLOCK_SIZE * i;

            const size_t SB_ROW = idx / BN;
            const size_t SB_COL = idx % BN;

            const size_t B_ROW = BK_OFFSET + SB_ROW;
            const size_t B_COL = BN_OFFSET + SB_COL;

            const size_t B_IDX = B_COL * K + B_ROW;
            SM_B[SB_ROW][SB_COL] = (B_ROW < K && B_COL < N) ? llaisys::utils::nvidia::cast<float>(B[B_IDX]) : float(0);
        }

        __syncthreads();

        // compute C tile
#pragma unroll
        for (size_t k = 0; k < BK; ++k) {

            // load register
            float REG_A[TM];
#pragma unroll
            for (size_t i = 0; i < (TM >> 2); ++ i) {
                FLOAT4(REG_A[i << 2]) = FLOAT4(SM_A[k][TM_OFFSET + (i << 2)]);
            }
            float REG_B[TN];
#pragma unroll
            for (size_t j = 0; j < (TN >> 2); ++ j) {
                FLOAT4(REG_B[(j << 2)]) = FLOAT4(SM_B[k][TN_OFFSET + (j << 2)]);
            }

            // compute
#pragma unroll
            for (size_t i = 0; i < TM; ++i) {
#pragma unroll
                for (size_t j = 0; j < TN; ++j) {
                    result[i][j] = fmaf(REG_A[i], REG_B[j], result[i][j]);
                }
            }
        }
    }

    // back to C tile
#pragma unroll
    for (size_t i = 0; i < TM; ++ i){
#pragma unroll
        for (size_t j = 0; j < TN; ++ j) {
            const size_t C_ROW = BM_OFFSET + TM_OFFSET + i;
            const size_t C_COL = BN_OFFSET + TN_OFFSET + j;

            if(C_ROW < M && C_COL < N){
                const size_t C_IDX = C_ROW * N + C_COL;
                float val = result[i][j] + ((bias != nullptr) ? llaisys::utils::nvidia::cast<float>(bias[C_COL]) : float(0));
                
                C[C_IDX] = llaisys::utils::nvidia::cast<T>(val);
            }
        }
    }
}

template <typename T>
void linear_launch(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                   const size_t m, const size_t n, const size_t k) {
    T *d_out = reinterpret_cast<T *>(out);
    const T *d_in = reinterpret_cast<const T *>(in);
    const T *d_weight = reinterpret_cast<const T *>(weight);
    const T *d_bias = reinterpret_cast<const T *>(bias);

    dim3 blockDim(BK_N, BK_M);
    dim3 gridDim((n + BN - 1) / BN, (m + BM - 1) / BM);

    linear_kernel<<<gridDim, blockDim>>>(d_out, d_in, d_weight, d_bias, m, n, k);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace

namespace llaisys::ops::nvidia {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias, llaisysDataType_t type,
            const size_t m, const size_t n, const size_t k) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_launch<float>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_BF16:
        return linear_launch<llaisys::bf16_t>(out, in, weight, bias, m, n, k);
    case LLAISYS_DTYPE_F16:
        return linear_launch<llaisys::fp16_t>(out, in, weight, bias, m, n, k);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}

} // namespace llaisys::ops::nvidia
