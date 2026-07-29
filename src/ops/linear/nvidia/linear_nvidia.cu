#include "linear_nvidia.cuh"

#include "../../../utils.hpp"
#include "utils/nvidia_utils.cuh"
#include <cublasLt.h>
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#define CUBLAS_CHECK(call)                                         \
    do {                                                           \
        cublasStatus_t err = (call);                               \
        if (err != CUBLAS_STATUS_SUCCESS) {                        \
            std::cerr << "cuBLAS error at "                        \
                      << __FILE__ << ":" << __LINE__               \
                      << " code=" << err << std::endl;             \
            std::exit(1);                                          \
        }                                                          \
    } while (0)

namespace {

// ── Cached execution context per shape ──────────────────────────

struct GemmContext {
    cublasLtMatmulDesc_t matmul_desc = nullptr;
    cublasLtMatrixLayout_t a_desc = nullptr; // in:  [M, K]
    cublasLtMatrixLayout_t b_desc = nullptr; // wgt: [N, K] (transposed to [K, N])
    cublasLtMatrixLayout_t c_desc = nullptr; // out: [M, N]
    cublasLtMatmulAlgo_t algo;
};

struct GemmKey {
    int64_t m, n, k;
    int cuda_dtype;

    bool operator==(const GemmKey &o) const {
        return m == o.m && n == o.n && k == o.k && cuda_dtype == o.cuda_dtype;
    }
};

struct GemmKeyHash {
    size_t operator()(const GemmKey &key) const {
        size_t h = static_cast<size_t>(key.m);
        h = h * 31 + static_cast<size_t>(key.n);
        h = h * 31 + static_cast<size_t>(key.k);
        h = h * 31 + static_cast<size_t>(key.cuda_dtype);
        return h;
    }
};

static std::unordered_map<GemmKey, GemmContext, GemmKeyHash> g_ctx_cache;
static std::mutex g_ctx_mutex;

// ── Global cublasLt handle + workspace ──────────────────────────

static cublasLtHandle_t g_lt_handle = nullptr;
static void *g_workspace = nullptr;
static size_t g_workspace_size = 0;

static void ensure_lt_handle() {
    if (g_lt_handle != nullptr) return;
    static std::once_flag flag;
    std::call_once(flag, []() {
        CUBLAS_CHECK(cublasLtCreate(&g_lt_handle));
        g_workspace_size = 4 * 1024 * 1024;
        cudaMalloc(&g_workspace, g_workspace_size);
    });
}

// ── Build cached context (called once per shape) ───────────────

static GemmContext build_context(int M, int N, int K, cudaDataType_t data_type,
                                 cublasComputeType_t compute_type) {
    GemmContext ctx;
    cublasLtOrder_t order_row = CUBLASLT_ORDER_ROW;
    cublasOperation_t trans_b = CUBLAS_OP_T;

    // matmul descriptor: compute=32F, scale=32F
    CUBLAS_CHECK(cublasLtMatmulDescCreate(&ctx.matmul_desc, compute_type, CUDA_R_32F));
    CUBLAS_CHECK(cublasLtMatmulDescSetAttribute(
        ctx.matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB,
        &trans_b, sizeof(trans_b)));

    // A (in):     [M, K]  row-major, ld=K
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &ctx.a_desc, data_type, M, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        ctx.a_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_row, sizeof(order_row)));

    // B (weight): [N, K]  row-major, ld=K,  trans→[K, N]
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &ctx.b_desc, data_type, N, K, K));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        ctx.b_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_row, sizeof(order_row)));

    // C (out):    [M, N]  row-major, ld=N
    CUBLAS_CHECK(cublasLtMatrixLayoutCreate(
        &ctx.c_desc, data_type, M, N, N));
    CUBLAS_CHECK(cublasLtMatrixLayoutSetAttribute(
        ctx.c_desc, CUBLASLT_MATRIX_LAYOUT_ORDER,
        &order_row, sizeof(order_row)));

    // Query best algorithm
    cublasLtMatmulPreference_t pref = nullptr;
    CUBLAS_CHECK(cublasLtMatmulPreferenceCreate(&pref));
    CUBLAS_CHECK(cublasLtMatmulPreferenceSetAttribute(
        pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_workspace_size, sizeof(g_workspace_size)));

    constexpr int MAX_ALGOS = 16;
    cublasLtMatmulHeuristicResult_t results[MAX_ALGOS];
    int returned = 0;
    CUBLAS_CHECK(cublasLtMatmulAlgoGetHeuristic(
        g_lt_handle, ctx.matmul_desc,
        ctx.a_desc, ctx.b_desc, ctx.c_desc, ctx.c_desc,
        pref, MAX_ALGOS, results, &returned));

    CUBLAS_CHECK(cublasLtMatmulPreferenceDestroy(pref));

    if (returned <= 0) {
        std::cerr << "cublasLtMatmulAlgoGetHeuristic returned 0 algos "
                  << "for M=" << M << " N=" << N << " K=" << K
                  << " dtype=" << (int)data_type << std::endl;
        std::exit(1);
    }

    ctx.algo = results[0].algo;
    return ctx;
}

// ── Per-call matmul (uses cached context) ───────────────────────

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

    ensure_lt_handle();

    GemmKey key{static_cast<int64_t>(M), static_cast<int64_t>(N),
                static_cast<int64_t>(K), static_cast<int>(data_type)};

    const GemmContext *ctx;
    {
        std::lock_guard<std::mutex> lock(g_ctx_mutex);
        auto it = g_ctx_cache.find(key);
        if (it == g_ctx_cache.end()) {
            auto [ins, _] = g_ctx_cache.emplace(
                key, build_context(static_cast<int>(M), static_cast<int>(N),
                                   static_cast<int>(K), data_type, compute_type));
            it = ins;
        }
        ctx = &it->second;
    }

    const float alpha = 1.0f;
    const float beta = 0.0f;

    CUBLAS_CHECK(cublasLtMatmul(
        g_lt_handle, ctx->matmul_desc,
        &alpha,
        const_cast<T *>(in), ctx->a_desc,
        const_cast<T *>(weight), ctx->b_desc,
        &beta,
        out, ctx->c_desc,
        out, ctx->c_desc,
        &ctx->algo,
        g_workspace, g_workspace_size,
        nullptr));
}

// ── add_bias kernel (unchanged) ─────────────────────────────────

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

template <typename T>
void linear_launch(std::byte *d_out, const std::byte *d_in, const std::byte *d_weight, const std::byte *d_bias,
                   const size_t m, const size_t n, const size_t k) {
    T *out = reinterpret_cast<T *>(d_out);
    const T *in = reinterpret_cast<const T *>(d_in);
    const T *weight = reinterpret_cast<const T *>(d_weight);
    const T *bias = reinterpret_cast<const T *>(d_bias);

    gemm_launch(out, in, weight, m, n, k);

    if (d_bias != nullptr) {
        dim3 blockDim(256);
        dim3 gridDim((m * n + blockDim.x - 1) / blockDim.x);
        add_bias_kernel<T><<<gridDim, blockDim>>>(out, bias, m, n);
        CUDA_CHECK(cudaGetLastError());
    }
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
