#include "gemv_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"

#include <cublasLt.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <vector>

#define CUBLASLT_CHECK(call)                                      \
    do {                                                          \
        const cublasStatus_t status = (call);                      \
        if (status != CUBLAS_STATUS_SUCCESS) {                     \
            std::cerr << "cuBLASLt error at "                      \
                      << __FILE__ << ":" << __LINE__ << std::endl; \
            std::exit(1);                                         \
        }                                                         \
    } while (0)

namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 4;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kLongWarpsPerBlock = 8;
constexpr int kLongThreadsPerBlock = kWarpSize * kLongWarpsPerBlock;
constexpr size_t kLongK = 4096;
constexpr size_t kLtWorkspaceBytes = 4 * 1024 * 1024;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset != 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <typename T, int M>
__device__ __forceinline__ void accumulate_vector(
    float (&acc)[M], const T *in, const T *weight, size_t k, size_t k0) {
    const float4 w = llaisys::utils::nvidia::load_4d(weight + k0);
#pragma unroll
    for (int row = 0; row < M; ++row) {
        const float4 x = llaisys::utils::nvidia::load_4d(
            in + static_cast<size_t>(row) * k + k0);
        acc[row] = fmaf(x.x, w.x, acc[row]);
        acc[row] = fmaf(x.y, w.y, acc[row]);
        acc[row] = fmaf(x.z, w.z, acc[row]);
        acc[row] = fmaf(x.w, w.w, acc[row]);
    }
}

template <typename T, int M>
__global__ void warp_per_column_kernel(
    T *__restrict__ out, const T *__restrict__ in,
    const T *__restrict__ weight, const T *__restrict__ bias,
    size_t n, size_t k) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    const size_t col = static_cast<size_t>(blockIdx.x) * kWarpsPerBlock
                       + static_cast<size_t>(warp);
    if (col >= n) {
        return;
    }

    float acc[M] = {};
    const T *weight_row = weight + col * k;
    for (size_t k0 = static_cast<size_t>(lane) * 4; k0 < k;
         k0 += static_cast<size_t>(kWarpSize) * 4) {
        accumulate_vector<T, M>(acc, in, weight_row, k, k0);
    }
#pragma unroll
    for (int row = 0; row < M; ++row) {
        acc[row] = warp_sum(acc[row]);
    }

    if (lane == 0) {
        const float bias_value = bias == nullptr
                                     ? 0.0f
                                     : llaisys::utils::nvidia::cast<float>(bias[col]);
#pragma unroll
        for (int row = 0; row < M; ++row) {
            out[static_cast<size_t>(row) * n + col]
                = llaisys::utils::nvidia::cast<T>(acc[row] + bias_value);
        }
    }
}

template <typename T, int M>
__global__ void block_per_column_kernel(
    T *__restrict__ out, const T *__restrict__ in,
    const T *__restrict__ weight, const T *__restrict__ bias,
    size_t n, size_t k) {
    const size_t col = blockIdx.x;
    const int tid = threadIdx.x;
    const int warp = tid / kWarpSize;
    const int lane = tid % kWarpSize;
    float acc[M] = {};
    const T *weight_row = weight + col * k;
    for (size_t k0 = static_cast<size_t>(tid) * 4; k0 < k;
         k0 += static_cast<size_t>(kLongThreadsPerBlock) * 4) {
        accumulate_vector<T, M>(acc, in, weight_row, k, k0);
    }

    __shared__ float warp_sums[M][kLongWarpsPerBlock];
#pragma unroll
    for (int row = 0; row < M; ++row) {
        acc[row] = warp_sum(acc[row]);
        if (lane == 0) {
            warp_sums[row][warp] = acc[row];
        }
    }
    __syncthreads();

    if (warp == 0) {
#pragma unroll
        for (int row = 0; row < M; ++row) {
            float value = lane < kLongWarpsPerBlock ? warp_sums[row][lane] : 0.0f;
            value = warp_sum(value);
            if (lane == 0) {
                if (bias != nullptr) {
                    value += llaisys::utils::nvidia::cast<float>(bias[col]);
                }
                out[static_cast<size_t>(row) * n + col]
                    = llaisys::utils::nvidia::cast<T>(value);
            }
        }
    }
}

struct LtPlan {
    cudaDataType_t dtype;
    size_t m;
    size_t n;
    size_t k;
    bool has_bias;
    cublasLtMatmulDesc_t operation = nullptr;
    cublasLtMatrixLayout_t weight_layout = nullptr;
    cublasLtMatrixLayout_t input_layout = nullptr;
    cublasLtMatrixLayout_t output_layout = nullptr;
    cublasLtMatmulAlgo_t algorithm{};

    LtPlan(cudaDataType_t dtype_, size_t m_, size_t n_, size_t k_, bool has_bias_,
           cublasLtHandle_t handle, void *workspace,
           void *out, const void *in, const void *weight, void *bias) :
        dtype(dtype_), m(m_), n(n_), k(k_), has_bias(has_bias_) {
        CUBLASLT_CHECK(cublasLtMatmulDescCreate(
            &operation, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        const cublasOperation_t trans_weight = CUBLAS_OP_T;
        const cublasOperation_t trans_input = CUBLAS_OP_N;
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSA,
            &trans_weight, sizeof(trans_weight)));
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_TRANSB,
            &trans_input, sizeof(trans_input)));
        if (has_bias) {
            const cublasLtEpilogue_t epilogue = CUBLASLT_EPILOGUE_BIAS;
            CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
                operation, CUBLASLT_MATMUL_DESC_EPILOGUE,
                &epilogue, sizeof(epilogue)));
            set_bias(bias);
        }

        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
            &weight_layout, dtype, k, n, static_cast<int64_t>(k)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
            &input_layout, dtype, k, m, static_cast<int64_t>(k)));
        CUBLASLT_CHECK(cublasLtMatrixLayoutCreate(
            &output_layout, dtype, n, m, static_cast<int64_t>(n)));

        cublasLtMatmulPreference_t preference = nullptr;
        CUBLASLT_CHECK(cublasLtMatmulPreferenceCreate(&preference));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceSetAttribute(
            preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &kLtWorkspaceBytes, sizeof(kLtWorkspaceBytes)));
        constexpr int max_algorithms = 8;
        std::array<cublasLtMatmulHeuristicResult_t, max_algorithms> results{};
        int returned = 0;
        CUBLASLT_CHECK(cublasLtMatmulAlgoGetHeuristic(
            handle, operation,
            weight_layout, input_layout, output_layout, output_layout,
            preference, max_algorithms, results.data(), &returned));
        CUBLASLT_CHECK(cublasLtMatmulPreferenceDestroy(preference));
        if (returned == 0) {
            throw std::runtime_error("cuBLASLt found no small-M algorithm");
        }

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cudaEvent_t start;
        cudaEvent_t end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));
        float best_ms = std::numeric_limits<float>::max();
        for (int index = 0; index < returned; ++index) {
            CUBLASLT_CHECK(cublasLtMatmul(
                handle, operation, &alpha,
                weight, weight_layout, in, input_layout, &beta,
                out, output_layout, out, output_layout,
                &results[index].algo, workspace, kLtWorkspaceBytes, nullptr));
            std::array<float, 3> samples{};
            for (float &sample : samples) {
                CUDA_CHECK(cudaEventRecord(start));
                CUBLASLT_CHECK(cublasLtMatmul(
                    handle, operation, &alpha,
                    weight, weight_layout, in, input_layout, &beta,
                    out, output_layout, out, output_layout,
                    &results[index].algo, workspace, kLtWorkspaceBytes, nullptr));
                CUDA_CHECK(cudaEventRecord(end));
                CUDA_CHECK(cudaEventSynchronize(end));
                CUDA_CHECK(cudaEventElapsedTime(&sample, start, end));
            }
            std::sort(samples.begin(), samples.end());
            if (samples[1] < best_ms) {
                best_ms = samples[1];
                algorithm = results[index].algo;
            }
        }
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }

    void set_bias(void *bias) {
        CUBLASLT_CHECK(cublasLtMatmulDescSetAttribute(
            operation, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
            &bias, sizeof(bias)));
    }
};

cublasLtHandle_t lt_handle() {
    static cublasLtHandle_t handle = [] {
        cublasLtHandle_t value = nullptr;
        CUBLASLT_CHECK(cublasLtCreate(&value));
        return value;
    }();
    return handle;
}

void *lt_workspace() {
    static void *workspace = [] {
        void *value = nullptr;
        CUDA_CHECK(cudaMalloc(&value, kLtWorkspaceBytes));
        return value;
    }();
    return workspace;
}

LtPlan &lt_plan(cudaDataType_t dtype, size_t m, size_t n, size_t k,
                bool has_bias, void *out, const void *in,
                const void *weight, void *bias) {
    static std::vector<std::unique_ptr<LtPlan>> plans;
    for (const auto &plan : plans) {
        if (plan->dtype == dtype && plan->m == m && plan->n == n
            && plan->k == k && plan->has_bias == has_bias) {
            if (has_bias) {
                plan->set_bias(bias);
            }
            return *plan;
        }
    }
    plans.emplace_back(std::make_unique<LtPlan>(
        dtype, m, n, k, has_bias, lt_handle(), lt_workspace(),
        out, in, weight, bias));
    return *plans.back();
}

template <typename T>
void launch_lt(T *out, const T *in, const T *weight, const T *bias,
               size_t m, size_t n, size_t k) {
    cudaDataType_t dtype;
    if constexpr (std::is_same_v<T, llaisys::fp16_t>) {
        dtype = CUDA_R_16F;
    } else {
        dtype = CUDA_R_16BF;
    }
    LtPlan &plan = lt_plan(
        dtype, m, n, k, bias != nullptr,
        out, in, weight, const_cast<T *>(bias));
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLASLT_CHECK(cublasLtMatmul(
        lt_handle(), plan.operation,
        &alpha,
        weight, plan.weight_layout,
        in, plan.input_layout,
        &beta,
        out, plan.output_layout,
        out, plan.output_layout,
        &plan.algorithm,
        lt_workspace(), kLtWorkspaceBytes,
        nullptr));
}

template <typename T, int M>
void launch_simt(T *out, const T *in, const T *weight, const T *bias,
                 size_t n, size_t k) {
    if (k <= kLongK) {
        const unsigned blocks = static_cast<unsigned>(
            (n + kWarpsPerBlock - 1) / kWarpsPerBlock);
        warp_per_column_kernel<T, M>
            <<<blocks, kThreadsPerBlock>>>(out, in, weight, bias, n, k);
    } else {
        block_per_column_kernel<T, M>
            <<<static_cast<unsigned>(n), kLongThreadsPerBlock>>>(
                out, in, weight, bias, n, k);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <typename T>
void launch(T *out, const T *in, const T *weight, const T *bias,
            size_t m, size_t n, size_t k) {
    if (m >= 5) {
        return launch_lt(out, in, weight, bias, m, n, k);
    }
#define LLAISYS_LAUNCH_M(value) \
    case value:                 \
        return launch_simt<T, value>(out, in, weight, bias, n, k)
    switch (m) {
        LLAISYS_LAUNCH_M(1);
        LLAISYS_LAUNCH_M(2);
        LLAISYS_LAUNCH_M(3);
        LLAISYS_LAUNCH_M(4);
    default:
        throw std::runtime_error("Small-M GEMV requires 1 <= M <= 15");
    }
#undef LLAISYS_LAUNCH_M
}

} // namespace

namespace llaisys::ops::nvidia {

bool gemv_supported(llaisysDataType_t dtype, size_t m, size_t, size_t k) {
    return (dtype == LLAISYS_DTYPE_F16 || dtype == LLAISYS_DTYPE_BF16)
           && m > 0 && m <= 15 && k > 0
           && (m > 4 || (k % 4) == 0);
}

void gemv(std::byte *out, const std::byte *in, const std::byte *weight,
          const std::byte *bias, llaisysDataType_t dtype,
          size_t m, size_t n, size_t k) {
    if (dtype == LLAISYS_DTYPE_F16) {
        return launch(
            reinterpret_cast<llaisys::fp16_t *>(out),
            reinterpret_cast<const llaisys::fp16_t *>(in),
            reinterpret_cast<const llaisys::fp16_t *>(weight),
            reinterpret_cast<const llaisys::fp16_t *>(bias), m, n, k);
    }
    if (dtype == LLAISYS_DTYPE_BF16) {
        return launch(
            reinterpret_cast<llaisys::bf16_t *>(out),
            reinterpret_cast<const llaisys::bf16_t *>(in),
            reinterpret_cast<const llaisys::bf16_t *>(weight),
            reinterpret_cast<const llaisys::bf16_t *>(bias), m, n, k);
    }
    throw std::runtime_error("Unsupported dtype for small-M GEMV");
}

} // namespace llaisys::ops::nvidia
