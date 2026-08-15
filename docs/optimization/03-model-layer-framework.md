# Model / Layer 框架

> 合并自 `2026-08-15-02/03/04` 与 `2026-08-16-05`。按时间线：骨架 → Qwen2 迁移 → Llama → 统一 API 终态。

## 1. 骨架期（§3，2026-08-15）

**目标：** 语义 Layer（Embedding / Attn / FFN）+ `Model` 基类 + Context（C 运行面 / D 上传面）并行落地。当时约定：**不接生产硬编码**、不做 P0/D0/D1 性能门禁。

| 面 | 内容 | 热路径 |
|---|---|---|
| **C** | `ModelContext`：meta / runtime / workspace + `AttnWeights[n]` / `FfnWeights[n]` + 全局 embed/norm | Layer 绑指针；`forward` 只用 `tensor_t` |
| **D** | `WeightRole` + `layer_idx` → `set_weight` | 仅加载/测试；热路径无 role 查找 |

当时验收（样例路径）：

| 测试 | 结果 |
|---|---|
| `llaisys-core-test` | PASS |
| `llaisys-model-layer-cpu-test` | PASS（embedding + FFN） |
| `llaisys-model-layer-nvidia-test` | PASS（层 0 PostAttn/PostFfn 与 legacy 对照；decode 对照；CPU 快照不 alias workspace） |
| 生产 `qwen2` / `python/` 相对 `86693ca` | 无改动 |

捕获：`capture_post_attn0/ffn0` 为残差后的 CPU 副本。骨架 commits：`a038de5` … `484ed0b`（交付报告）。

> 布局已演进为 `src/models/core/`、`src/layers/`、命名空间 `llaisys::core`（见 §4）。

## 2. Qwen2 迁入 Layer（§4，2026-08-16）

**目标：** 生产默认 Layer；legacy 对照；P0/D0/D1 相对 legacy ≤3%。

| Task | Commit | 说明 |
|---|---|---|
| 1 | `70898c4` | `prepare_*` 上提 `Model` |
| 2 | `1ee414f` | 双路径（默认仍 legacy） |
| 3 | `f4edc23` | greedy 硬门禁（42 tokens 一致） |
| 4 | `933e3b5` | Python → Context（当时 Scheme A） |
| 5 | `940ff20` | 默认 Layer |
| 6 | （报告） | P0/D0/D1 性能门禁 |

硬门禁：`test_qwen2_layer_parity.py` legacy vs layer greedy **42 tokens 完全一致**。

性能门禁（`migration_perf_gate.py`，median `wall_per_output_token` ms/tok，3 次，`gate_status: passed`；原始 `test/results/migration_perf_gate_20260816_020601.json`）：

| Shape | legacy median | layer median | Δ | ≤3% |
|---|---:|---:|---:|---|
| P0 (256→1) | 47.364 | 47.234 | -0.27% | PASS |
| D0 (1→128) | 16.381 | 16.041 | -2.08% | PASS |
| D1 (1024→128) | 18.302 | 18.487 | +1.01% | PASS |

**结论：** 保持默认 Layer。

历史回滚方式 `LLAISYS_QWEN2_LAYER_FORWARD=0` **已失效**（`forward_legacy` 已删）。

## 3. Llama 纵向切片

**目标：** 同框架接入 Llama；不改 Scheduler / Sequence / BlockManager。

| 当时做法 | 终态（见下节） |
|---|---|
| env `LLAISYS_LAYER_STACK=llama` + `run_llama_layer_stack` | **已删** |
| 借用 Qwen2 C API / Weights | `Llama : Qwen2` + `Create(LLAMA)` + `SetWeight` |

限制：**未实现** Llama-3 `rope_scaling`；`rope_theta=500000` 标准 RoPE。

冒烟（NVIDIA，4 step）：`generated_tokens=4`，解码 `Hello! How can` → **PASS**（RTX 4060 8GB）。

## 4. 统一 Model C API 终态（2026-08-16）

| 原约定 | 现状 |
|---|---|
| 共存 A / 硬编码 | 仅 Layer 组装 |
| `llaisysQwen2*` + Weights 结构体 | `llaisysModel*` + `SetWeight(role, layer, tensor)` |
| `LLAISYS_LAYER_STACK` / `run_*_stack` | 删除；`Qwen2`/`Llama::forward` 内组装 |
| `llaisys::framework` | `llaisys::core` |
| Scheme A sync | 删除 |

目录：

```text
include/llaisys/models/model.h
src/llaisys/models/model.cc
src/models/core/{model,model_context}.*
src/models/qwen2/*    src/models/llama/*
src/layers/{layer.hpp,qwen2/*,llama/*}
python/llaisys/libllaisys/models/model.py
python/llaisys/models/{qwen2,llama}.py
```

验证：symbols / request API / weight_map / layer-cpu / `test_infer` 短步 / Llama smoke — PASS。

非目标：rope_scaling；改调度/KV 生命周期；NF4/KV 量化（后续阶段）。

相关提交：`6d94873`（统一 API + core）、`766589e`（交付收口）。
