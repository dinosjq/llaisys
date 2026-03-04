#include "linear_nvidia.cuh"

#include "utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <type_traits>

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t err = (call);                               \
        if (err != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLASLt error at "                      \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace {
/**
 * TODO:
 * 调用 cublas 的 api 实现 linear
 * 还有自己实现 linear 且做性能对比
 */

// // blockDim
// static constexpr size_t BLOCK_SIZE = 256;
// static constexpr size_t BK_M = 16;
// static constexpr size_t BK_N = 16;

// // smem size
// static constexpr size_t SMEM_SIZE = 1024;
// static constexpr size_t LOAD_COUNT = 4;

// // thread
// static constexpr size_t TM = 8;
// static constexpr size_t TN = 8;

// // block tiling
// static constexpr size_t BM = 128;
// static constexpr size_t BN = 128;
// static constexpr size_t BK = 8;

// /**
//  * DIM: C: [ M x N ], A: [ M x K ], B: [ N x K ], BIAS: [ N ]
//  * GEMM: 2D-block tiling, shared memory, thread tiling, 向量化访存, 双缓冲 ...
//  *
//  *  ? 不是很会解决 bank conflict
//  *
//  * 最后写法优化 彻底丧失可读性
//  */

// template <typename T>
// __global__ void linear_kernel(T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B, const T *__restrict__ bias,
//                               const size_t M, const size_t N, const size_t K) {
//     // smem
//     __shared__ float SM_A[2][BK][BM];
//     __shared__ float SM_B[2][BK][BN];

//     // block
//     const size_t BM_OFFSET = blockIdx.y << 7; // * BM;
//     const size_t BN_OFFSET = blockIdx.x << 7; // * BN;

//     // thread
//     const size_t tid = threadIdx.y * blockDim.x + threadIdx.x;
//     const size_t TM_OFFSET = threadIdx.y << 3; // * TM;
//     const size_t TN_OFFSET = threadIdx.x << 3; // * TN;

//     // compute result
//     float result[TM][TN] = {{float(0)}};

//     float REG_A[2][TM];
//     float REG_B[2][TN];

//     int buffer_id = 0;

//     // pre-read A and B tile smem 0
// #pragma unroll
//     for (size_t i = 0; i < LOAD_COUNT; ++i) {
//         size_t idx = tid + (i << 8);

//         const size_t SA_ROW = idx & 127;
//         const size_t SA_COL = idx >> 7;

//         const size_t A_ROW = BM_OFFSET + SA_ROW;
//         const size_t A_COL = 0 + SA_COL;

//         const size_t A_IDX = A_ROW * K + A_COL;
//         SM_A[0][SA_COL][SA_ROW] = (A_ROW < M && A_COL < K) ? llaisys::utils::nvidia::cast<float>(A[A_IDX]) : float(0);
//     }

//     {
//         const size_t SB_ROW = (tid & 1) << 2;
//         const size_t SB_COL = tid >> 1;

//         const size_t B_ROW = BN_OFFSET + SB_COL;
//         const size_t B_COL = 0 + SB_ROW;

//         const size_t B_IDX = B_ROW * K + B_COL;

//         if (B_ROW < N && B_COL + 3 < K) {
//             float4 flo4 = llaisys::utils::nvidia::load_4d<T>(B + B_IDX);
//             SM_B[0][SB_ROW | 0][SB_COL] = flo4.x;
//             SM_B[0][SB_ROW | 1][SB_COL] = flo4.y;
//             SM_B[0][SB_ROW | 2][SB_COL] = flo4.z;
//             SM_B[0][SB_ROW | 3][SB_COL] = flo4.w;
//         } else {
// #pragma unroll
//             for (int i = 0; i < 4; ++i) {
//                 SM_B[0][SB_ROW + i][SB_COL] = (B_ROW < N && B_COL + i < K) ? llaisys::utils::nvidia::cast<float>(B[B_IDX + i]) : float(0);
//             }
//         }
//     }

//     __syncthreads();

//     for (size_t BK_OFFSET = BK; BK_OFFSET < K + BK; BK_OFFSET += BK) {
//         // compute and load
// #pragma unroll
//         for (size_t k = 0; k < BK + 1; ++k) {
//             // compute k - 1
//             if (k > 0) {
// #pragma unroll
//                 for (size_t i = 0; i < TM; ++i) {
// #pragma unroll
//                     for (size_t j = 0; j < TN; ++j) {
//                         result[i][j] = fmaf(REG_A[(k & 1) ^ 1][i], REG_B[(k & 1) ^ 1][j], result[i][j]);
//                     }
//                 }
//             }

//             // load register k
//             if (k < BK) {
// #pragma unroll
//                 for (size_t i = 0; i < (TM >> 2); ++i) {
//                     FLOAT4(REG_A[k & 1][i << 2]) = FLOAT4(SM_A[buffer_id][k][TM_OFFSET + (i << 2)]);
//                 }
// #pragma unroll
//                 for (size_t j = 0; j < (TN >> 2); ++j) {
//                     FLOAT4(REG_B[k & 1][(j << 2)]) = FLOAT4(SM_B[buffer_id][k][TN_OFFSET + (j << 2)]);
//                 }
//             }
//         }

//         // load A and B tile
//         if (BK_OFFSET < K) {
// #pragma unroll
//             for (size_t i = 0; i < LOAD_COUNT; ++i) {
//                 size_t idx = tid + (i << 8);

//                 const size_t SA_ROW = idx & 127;
//                 const size_t SA_COL = idx >> 7;

//                 const size_t A_ROW = BM_OFFSET + SA_ROW;
//                 const size_t A_COL = BK_OFFSET + SA_COL;

//                 const size_t A_IDX = A_ROW * K + A_COL;
//                 SM_A[buffer_id ^ 1][SA_COL][SA_ROW] = (A_ROW < M && A_COL < K) ? llaisys::utils::nvidia::cast<float>(A[A_IDX]) : float(0);
//             }

//             {
//                 const size_t SB_ROW = (tid & 1) << 2;
//                 const size_t SB_COL = tid >> 1;

//                 const size_t B_ROW = BN_OFFSET + SB_COL;
//                 const size_t B_COL = BK_OFFSET + SB_ROW;

//                 const size_t B_IDX = B_ROW * K + B_COL;

//                 if (B_ROW < N && B_COL + 3 < K) {
//                     float4 flo4 = llaisys::utils::nvidia::load_4d<T>(B + B_IDX);
//                     SM_B[buffer_id ^ 1][SB_ROW | 0][SB_COL] = flo4.x;
//                     SM_B[buffer_id ^ 1][SB_ROW | 1][SB_COL] = flo4.y;
//                     SM_B[buffer_id ^ 1][SB_ROW | 2][SB_COL] = flo4.z;
//                     SM_B[buffer_id ^ 1][SB_ROW | 3][SB_COL] = flo4.w;
//                 } else {
// #pragma unroll
//                     for (int i = 0; i < 4; ++i) {
//                         SM_B[buffer_id ^ 1][SB_ROW | i][SB_COL] = (B_ROW < N && B_COL + i < K) ? llaisys::utils::nvidia::cast<float>(B[B_IDX + i]) : float(0);
//                     }
//                 }
//             }

//             __syncthreads();
//         }

//         buffer_id ^= 1;
//     }

//     // back to C tile
// #pragma unroll
//     for (size_t i = 0; i < TM; ++i) {
//         const size_t C_ROW = BM_OFFSET + TM_OFFSET + i;

//         if (C_ROW >= M) {
//             continue;
//         }

// #pragma unroll
//         for (size_t j = 0; j < TN; ++j) {
//             const size_t C_COL = BN_OFFSET + TN_OFFSET + j;

//             if (C_COL >= N) {
//                 continue;
//             }

//             const size_t C_IDX = C_ROW * N + C_COL;
//             float val = result[i][j] + ((bias != nullptr) ? llaisys::utils::nvidia::cast<float>(bias[C_COL]) : float(0));

//             C[C_IDX] = llaisys::utils::nvidia::cast<T>(val);
//         }
//     }
// }

/* ---------------------------------------- */

template <typename T>
void gemm_launch(T *out, const T *in, const T *weight,
                 const size_t M, const size_t N, const size_t K) {
    cudaDataType_t data_type;
    cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;

    if constexpr (std::is_same_v<T, float>) {
        data_type = CUDA_R_32F;
    } else if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        data_type = CUDA_R_16F;
    } else if constexpr (std::is_same_v<T, llaisys::bf16_t>) {
        data_type = CUDA_R_16BF;
    } else {
        throw std::runtime_error("Unsupported data type for cuBLAS GEMM");
    }

    // Reuse a thread-local cuBLAS handle to avoid create/destroy overhead.
    static thread_local cublasHandle_t handle = nullptr;
    static thread_local bool handle_inited = false;
    
    if (!handle_inited) {
        CUBLAS_CHECK(cublasCreate(&handle));
        // Enable tensor cores for fp16/bf16
        if (data_type == CUDA_R_16F || data_type == CUDA_R_16BF) {
            cublasStatus_t s = cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
            (void)s;
        }
        handle_inited = true;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,         // weight 做转置
        CUBLAS_OP_N,         // in 不转置
        static_cast<int>(N), // m
        static_cast<int>(M), // n
        static_cast<int>(K), // k
        &alpha,
        reinterpret_cast<const void *>(weight), data_type, static_cast<int>(K), // lda for weight
        reinterpret_cast<const void *>(in), data_type, static_cast<int>(K),     // ldb for in
        &beta,
        reinterpret_cast<void *>(out), data_type, static_cast<int>(N), // ldc
        compute_type,
        CUBLAS_GEMM_DEFAULT));

    // Do not destroy handle here (reused across calls). Check for errors.
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
__global__ void add_bias_kernel(T *out, const T *bias, const size_t M, const size_t N) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t stride = static_cast<size_t>(gridDim.x) * static_cast<size_t>(blockDim.x);
    const size_t total = M * N;

    for (size_t idx = tid; idx < total; idx += stride) {
        const size_t col = idx % N;
        const float v = llaisys::utils::nvidia::cast<float>(out[idx]) + llaisys::utils::nvidia::cast<float>(bias[col]);
        out[idx] = llaisys::utils::nvidia::cast<T>(v);
    }
}

/* ---------------------------------------- */

/**
 * GEMM:
 * OUT:[M, N], IN:[M, K], WEIGHT:[N, K]
 */

/* ----- shared memory ----- */
// constexpr static int SMEM_SIZE = 1024;
// constexpr static int LOAD_COUNT = 4;

// /* ----- A B C tile ----- */
// // tile-A : [BM X BK]
// // tile-B : [BK X BN]
// // tile-C : [BM X BN]
// constexpr static int BM = 128;
// constexpr static int BN = 128;
// constexpr static int BK = 8;

// /* ----- A B C tile threads ----- */
// constexpr static int A_BLK_M = 32;
// constexpr static int A_BLK_K = 8;
// constexpr static int B_BLK_N = 32;
// constexpr static int B_BLK_K = 8;
// constexpr static int C_BLK_M = 16;
// constexpr static int C_BLK_N = 16;

// /* ----- thread compute ----- */
// constexpr static int TM = 8;
// constexpr static int TN = 8;

// /* ----- C tile warp threads ----- */
// constexpr static int C_WRP_M = 4;
// constexpr static int C_WRP_N = 8;

// /* store C:[M X N], A:[M X K], B:[N X K] */
// template <typename T>
// __global__ void gemm_kernel(T *__restrict__ C, const T *__restrict__ A, const T *__restrict__ B, const size_t M, const size_t N, const size_t K) {
//     __shared__ float SA[2][BK][BM];
//     __shared__ float SB[2][BK][BN];

//     // basic info for thread
//     const int tid = threadIdx.x;
//     const int warp_id = tid >> 5;
//     const int lane_id = tid & 31;

//     // offset for block
//     const int GRID_M = blockIdx.y * BM;
//     const int GRID_N = blockIdx.x * BN;

//     // offset for C
//     const int warp_y = warp_id / (C_BLK_N / C_WRP_N);
//     const int warp_x = warp_id & (C_BLK_N / C_WRP_N - 1);

//     const int lane_y = (lane_id & 1) + ((lane_id >> 4) << 1);
//     const int lane_x = (lane_id & 15) >> 1;

//     const int C_THR_M = warp_y * C_WRP_M + lane_y;
//     const int C_THR_N = warp_x * C_WRP_N + lane_x;

//     // offset for A
//     const int A_THR_M = tid / A_BLK_K;
//     const int A_THR_K = tid & (A_BLK_K - 1);

//     // offset for B
//     const int B_THR_N = tid / B_BLK_K;
//     const int B_THR_K = tid & (B_BLK_K - 1);

//     // registers
//     float acc[TM][TN] = {{0.0f}};
//     float A_REG[2][TM];
//     float B_REG[2][TN];

//     // pre-load
//     int buffer_id = 0;

//     // load A tile
// #pragma unroll
//     for (int i = 0; i < BM; i += A_BLK_M) {
//         const int row = i + A_THR_M + GRID_M;
//         SA[0][A_THR_K][(i + A_THR_M) ^ (A_THR_K << 2)] = (row < M && A_THR_K < K) ? llaisys::utils::nvidia::cast<float>(A[row * K + A_THR_K]) : 0.0f;
//     }

//     // load B tile
// #pragma unroll
//     for (int i = 0; i < BN; i += B_BLK_N) {
//         const int row = i + B_THR_N + GRID_N;
//         SB[0][B_THR_K][(i + B_THR_N) ^ (B_THR_K << 2)] = (row < N && B_THR_K < K) ? llaisys::utils::nvidia::cast<float>(B[row * K + B_THR_K]) : 0.0f;
//     }

//     __syncthreads();

//     // compute and load
//     for (int GRID_K = BK; GRID_K < K + BK; GRID_K += BK) {
// #pragma unroll
//         for (int k = 0; k < BK + 1; ++k) {
//             if (k > 0) {
//                 // compute C tile
// #pragma unroll
//                 for (int i = 0; i < TM; ++i) {
// #pragma unroll
//                     for (int j = 0; j < TN; ++j) {
//                         acc[i][j] += A_REG[(k - 1) & 1][i] * B_REG[(k - 1) & 1][j];
//                     }
//                 }
//             }
//             if (k < BK) {
//                 // load A register
// #pragma unroll
//                 for (int i = 0; i < (TM >> 2); ++i) {
//                     const int row = (C_THR_M + i * C_BLK_M) << 2;
//                     FLOAT4(A_REG[k & 1][i << 2]) = FLOAT4(SA[buffer_id][k][row ^ (k << 2)]);
//                 }
//                 // load B register
// #pragma unroll
//                 for (int j = 0; j < (TN >> 2); ++j) {
//                     const int col = (C_THR_N + j * C_BLK_N) << 2;
//                     FLOAT4(B_REG[k & 1][j << 2]) = FLOAT4(SB[buffer_id][k][col ^ (k << 2)]);
//                 }
//             }
//         }

//         if (GRID_K < K) {
//             // load A tile
//             {
//                 const int col = A_THR_K + GRID_K;
// #pragma unroll
//                 for (int i = 0; i < BM; i += A_BLK_M) {
//                     const int row = i + A_THR_M + GRID_M;
//                     SA[buffer_id ^ 1][A_THR_K][(i + A_THR_M) ^ (A_THR_K << 2)] = (row < M && col < K) ? llaisys::utils::nvidia::cast<float>(A[row * K + col]) : 0.0f;
//                 }
//             }

//             // load B tile
//             {
//                 const int col = B_THR_K + GRID_K;
// #pragma unroll
//                 for (int i = 0; i < BN; i += B_BLK_N) {
//                     const int row = i + B_THR_N + GRID_N;
//                     SB[buffer_id ^ 1][B_THR_K][(i + B_THR_N) ^ (B_THR_K << 2)] = (row < N && col < K) ? llaisys::utils::nvidia::cast<float>(B[row * K + col]) : 0.0f;
//                 }
//             }

//             __syncthreads();
//         }

//         buffer_id ^= 1;
//     }

//     // load back to C
// #pragma unroll
//     for (int i = 0; i < TM; ++i) {
//         const int row = (C_THR_M << 2) + ((i >> 2) * (C_BLK_M << 2) + (i & 3)) + GRID_M;
// #pragma unroll
//         for (int j = 0; j < TN; ++j) {
//             const int col = (C_THR_N << 2) + ((j >> 2) * (C_BLK_N << 2) + (j & 3)) + GRID_N;
//             if (row < M && col < N) {
//                 C[row * N + col] = llaisys::utils::nvidia::cast<T>(acc[i][j]);
//             }
//         }
//     }
// }

template <typename T>
void linear_launch(std::byte *d_out, const std::byte *d_in, const std::byte *d_weight, const std::byte *d_bias,
                   const size_t m, const size_t n, const size_t k) {
    T *out = reinterpret_cast<T *>(d_out);
    const T *in = reinterpret_cast<const T *>(d_in);
    const T *weight = reinterpret_cast<const T *>(d_weight);
    const T *bias = reinterpret_cast<const T *>(d_bias);

    /* ---------------------------------------- */

    /** END: 自定义linear调用 */
    // 最速: 7.45s 平均: 8s
    // dim3 blockDim(16, 16);
    // dim3 gridDim((n + 127) >> 7, (m + 127) >> 7);
    // linear_kernel<<<gridDim, blockDim>>>(out, in, weight, bias, m, n, k);

    /* ---------------------------------------- */

    /** END: cublas调用gemm + 自定义转置加和核函数 */
    // cublas: gemm
    // 最速: 1.86s 平均: 2s
    gemm_launch(out, in, weight, m, n, k);

    /* ---------------------------------------- */

    /** END: 自定义GEMM */
    // 虽然自定义gemm性能测试比linear要好，但是实际使用慢的要死
    // dim3 blockDim(256);
    // dim3 gridDim((n + 127) >> 7, (m + 127) >> 7);
    // gemm_kernel<<<gridDim, blockDim>>>(out, in, weight, m, n, k);

    // add bias
    if (d_bias != nullptr) {
        dim3 blockDim(256);
        dim3 gridDim((m * n + blockDim.x - 1) / blockDim.x);
        add_bias_kernel<T><<<gridDim, blockDim>>>(out, bias, m, n);
        CUDA_CHECK(cudaGetLastError());
    }

    /* ---------------------------------------- */
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
