# 统一 Model C API + core 命名空间 — 交付报告

**日期:** 2026-08-16  
**分支:** `main`（本地）  
**规格:** `docs/superpowers/specs/2026-08-16-unified-model-api-design.md`  
**收口计划:** `docs/superpowers/plans/2026-08-16-phase3-4-closeout.md`

## 目标

在 §3 骨架与 §4 Qwen2/Llama 迁移之后，收口对外 ABI 与目录命名，使 API 与后端 `Model`/`WeightRole`/`Layer` 一致。

## 相对 §3 / §4 原交付的演进

| 原交付约定 | 本轮状态 |
|---|---|
| 共存策略 A / 硬编码保留 | 已删除 `forward_legacy`；生产仅 Layer 组装 |
| `llaisysQwen2*` + `LlaisysQwen2Weights` | 删除；改为 `llaisysModel*` + `SetWeight` |
| `LLAISYS_LAYER_STACK` / `run_*_stack` | 删除；`Qwen2`/`Llama::forward` 内显式组装 |
| `llaisys::framework` | 重命名为 `llaisys::core`（对齐 `src/models/core/`） |
| Scheme A legacy sync | 删除；权重只经 `SetWeight` → `Model::set_weight` |
| Llama 借 Qwen2 + env | `class Llama : public Qwen2`；`Create(LLAMA)` |

## 目录（当前）

```text
include/llaisys/models/model.h
src/llaisys/models/model.cc
src/models/core/{model,model_context}.*
src/models/qwen2/*
src/models/llama/*
src/layers/{layer.hpp,qwen2/*,llama/*}
python/llaisys/libllaisys/models/model.py
python/llaisys/models/{qwen2,llama}.py
```

## 验证

| 项 | 结果 |
|---|---|
| `xmake` 共享库 | PASS |
| `test_model_api_symbols.py` | PASS（新符号在、旧 `llaisysQwen2*` 不在） |
| `test_qwen2_request_api.py` / weight_map / lifetime | PASS |
| `llaisys-model-layer-cpu-test` | PASS |
| `test_infer.py`（Qwen2 NVIDIA，短步） | PASS |
| `test_llama_smoke.py`（NVIDIA，4 token） | PASS（`Hello! How can`） |

## 非目标（仍不做）

- Llama-3 `rope_scaling`
- 改 Scheduler / Sequence / BlockManager
- NF4 / KV 量化等后续优化阶段

## 回滚

回退本轮 commits；恢复 `qwen2.h` / Scheme A 仅作考古，不推荐。

## 与总计划对应

- 优化总计划 **§3** 骨架：历史交付见 `2026-08-15-02-*.md`；本文件记录其后 ABI/命名收口。
- 优化总计划 **§4** 迁移 + Llama：历史门禁见 `03`/`04`；本文件记录默认路径与 API 终态。
