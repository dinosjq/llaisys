# linear 算子 M 阈值分派优化设计

## 背景

nsys 分析发现当前 `linear` 算子的性能瓶颈分布：

| 阶段 | M 值 | cuBLAS kernel | 平均耗时 | 占 LLAISYS 时间 |
|------|------|---------------|----------|-----------------|
| Prefill | 23 | `ampere_bf16_s1688gemm` (Tensor Core) | 275 us | 51.7% |
| Decode | 1 | `internal::gemvx::kernel` (GEMV) | 220 us | **63.2%** |

**核心问题**：`cublasGemmEx` + `CUBLAS_GEMM_DEFAULT` 对 M=1 的 decode 场景退化为 GEMV kernel，而 GEMV 因缺乏 Tensor Core 分块优化成为最大瓶颈。

**优化目标**：Decode linear 占比从 63.2% 降至 40% 以下，Prefill 不退化。

## 设计

### 架构：M 阈值分派

```
linear(out, in, weight, bias, M, N, K)
  │
  ├─ M > 4  ──► gemm_cublasGemmEx()   ← 保持现有 cublasGemmEx 调用
  │
  └─ M ≤ 4  ──► gemv_cublasLt()       ← 新增 cublasLtMatmul 列主序路径
                     │
                     └─ (未来) gemv_custom()  ← 预留接口
```

**阈值选择 M=4**：Prefill 最小 batch 约 23 tokens，Decode 恒为 1。阈值 4
给未来的 chunked prefill（M=2~4）留空间。业界惯例（vLLM、TensorRT-LLM）也在此阈值切分。

### Prefill 路径（M > 4）：不变

```cpp
// 完全保留现有实现：
cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
             N, M, K, &alpha,
             weight, data_type, K,
             in, data_type, K,
             &beta, out, data_type, N,
             compute_type, CUBLAS_GEMM_DEFAULT);
```

已验证该路径触达 Tensor Core GEMM（`ampere_bf16_s1688gemm`），无需修改。

### Decode 路径（M ≤ 4）：cublasLtMatmul

#### 矩阵布局（列主序）

cuBLAS 内部以列主序（column-major）为最优路径。使用默认列主序
（**不**设 `CUBLASLT_ORDER_ROW`），匹配 `cublasGemmEx` 的内部约定：

| 描述符 | 列主序尺寸 | ld | Transpose | op 结果 |
|--------|-----------|-----|-----------|---------|
| A (in) | [K, M] | K | `CUBLAS_OP_T` | [M, K] |
| B (weight) | [K, N] | K | `CUBLAS_OP_N` | [K, N] = weight^T |
| C (out) | [N, M] | N | — | [N, M] |

计算：op(A)[M, K] × op(B)[K, N] = C[N, M]，即 row-major 下的 out[M, N]。

#### 上下文缓存

为每个唯一的 (M, N, K, dtype) 四元组缓存完整的执行上下文：

```cpp
struct GemmContext {
    cublasLtMatmulDesc_t matmul_desc;   // 含 TRANSB 属性
    cublasLtMatrixLayout_t a_layout;    // A 矩阵布局
    cublasLtMatrixLayout_t b_layout;    // B 矩阵布局
    cublasLtMatrixLayout_t c_layout;    // C 矩阵布局
    cublasLtMatmulAlgo_t algo;          // 最优算法
};

// 全局缓存（按 (M,N,K,dtype) 索引）
static std::unordered_map<GemmKey, GemmContext> g_ctx_cache;
static std::mutex g_ctx_mutex;
```

首次调用时调用 `cublasLtMatmulAlgoGetHeuristic()` 查询 16 个候选算法，
选最优者缓存。后续调用直接查表、零额外开销。

#### 执行流程

```
gemv_cublasLt(out, in, weight, M, N, K):
  1. 查 g_ctx_cache
  2. 若无 → build_context() → 缓存
  3. cublasLtMatmul(handle, ctx.matmul_desc,
       &alpha, in, ctx.a_layout, weight, ctx.b_layout,
       &beta, out, ctx.c_layout, out, ctx.c_layout,
       &ctx.algo, workspace, 0)
```

### Bias 处理

当前 bias 由独立 `add_bias_kernel` launch 完成，占 0.3%。暂不融合。
未来与 `gemv_custom()` 一并处理，直接在 GEMV kernel 内累加 bias。

### 自定义 GEMV 预留接口

```cpp
template <typename T>
void gemv_custom(T *out, const T *in, const T *weight, const T *bias,
                 size_t M, size_t N, size_t K);
```

将来实现要点：
- 一个 warp 负责一行输出（128 元素），向量化访存（`copy_4d`）
- 权重按 K 维分片，shared memory 缓冲
- 直接融合 bias add，消除额外的 kernel launch

### 文件变更

| 文件 | 变更 |
|------|------|
| `src/ops/linear/nvidia/linear_nvidia.cu` | 重写：M 阈值分派 + `gemv_cublasLt()` + `GemmContext` 缓存 |
| `src/ops/linear/nvidia/backup.txt` | 更新为改动前版本的备份 |

`op.cpp`、C API、Python 层不变 —— 接口签名完全兼容。

### 全局状态管理

- `g_lt_handle`：全局 `cublasLtHandle_t`，`std::call_once` 初始化
- `g_workspace`：4 MiB GPU workspace，供 cublasLt split-K 使用
- `g_ctx_cache`：`GemmContext` 缓存，在进程退出时随静态变量析构

## 验证计划

1. **编译**：`xmake && xmake install`
2. **算子正确性**：`python test/ops/linear.py --device nvidia`
3. **端到端推理正确性**：`python test/test_infer.py --device nvidia --model <model> --test --max_steps 16`
4. **nsys profile**：
   ```bash
   nsys profile -o results/nsys_linear_v3 --force-overwrite true \
     python test/test_infer.py --device nvidia --model <model> --test --max_steps 16
   ```
5. **性能对比**：`python test/profile/nsys_report.py results/nsys_linear_v3.sqlite`
   - Decode `linear (gemm)` 总耗时占比目标：**< 40%**（基线 63.2%）
   - Prefill `linear (gemm)` 平均耗时目标：**不退化**（基线 275 us）

## 风险与回退

- **风险**：cublasLtMatmul 列主序对 M=1 的算法选择可能不比 cublasGemmEx 好
- **缓解**：如果 nsys 显示无改善，回退到 `cublasGemmEx` 同时保留 M 阈值分派结构，为后续自定义 GEMV 铺路
- **回退方案**：一行 diff —— 将 `gemv_cublasLt()` 调用改回 `gemm_cublasGemmEx()`
