# Model 基类对齐 + 去硬编码 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 按修订规格把 scheduler/KV/worker/request API 上提到 `Model`，目录改为 `layers/` + `models/core` + `models/qwen2/` + `sampling/`，删除硬编码 `forward_legacy`。

**Architecture:** `Qwen2 : framework::Model`；`forward` 虚函数走 Layer stack；`BatchPack` 取代 `Qwen2Pack`；语义层合并为每模型一个 hpp/cpp。

**Tech Stack:** C++17、现有 ops、xmake、ctypes Python 桥不变。

## Global Constraints

- 风格：对齐现有 qwen2（中文步骤注释、`this->`、成员 `_` 前缀）
- 不改 Scheduler/Sequence/BlockManager 语义
- C API 仍为 `llaisysQwen2*`；Python 对外不变
- 不推远端；可一个或少量 commit

---

## File Structure

| Path | Responsibility |
|---|---|
| `src/models/core/model_context.hpp` | Context + BatchPack + WeightRole |
| `src/models/core/model.hpp/.cpp` | Model 基类运行时 |
| `src/models/qwen2/qwen2.hpp/.cpp` | Qwen2 子类 |
| `src/layers/layer.hpp` | Layer 基类 |
| `src/layers/qwen2/qwen2.hpp/.cpp` | Qwen2 语义层 + 组装 |
| `src/layers/llama/llama.hpp/.cpp` | Llama 语义层 + 组装 |
| `src/sampling/sampling.hpp/.cpp` | 采样 |
| `xmake.lua` | 源路径 |

---

### Task 1: core 类型合并（BatchPack / WeightRole）

**Files:** `model_context.hpp`；删 `qwen2_pack.hpp`、`weight_role.hpp`

- [ ] Step 1: 把 `BatchPack`、`WeightRole` 并入 `model_context.hpp`
- [ ] Step 2: `Model::prepare_*` 返回 `BatchPack`；更新 `model.cpp`

---

### Task 2: Model 基类上提运行时

**Files:** `model.hpp/.cpp`

- [ ] Step 1: 迁入 scheduler/KV/worker/request/start/stop/submit/await/abort/release
- [ ] Step 2: `virtual forward(BatchPack&, ...) = 0`；worker 调虚函数
- [ ] Step 3: `ModelRequest`（字段同原 `Qwen2Request`）；保留 `using qwen2_request_t = model_request_t` 兼容

---

### Task 3: 目录与层文件合并

**Files:** `models/qwen2/*`、`layers/qwen2/*`、`layers/llama/*`、`sampling/*`

- [ ] Step 1: 合并 emb/attn/ffn/stack → 单 hpp/cpp；修 include
- [ ] Step 2: 移动 `models/qwen2`、`sampling`；更新 xmake

---

### Task 4: Qwen2 子类 + 删硬编码

**Files:** `models/qwen2/qwen2.*`、测试、Python 文档字符串

- [ ] Step 1: `class Qwen2 : public framework::Model`；只留构造/权重/forward/sample
- [ ] Step 2: 删除 `forward_legacy` 与 `LLAISYS_QWEN2_LAYER_FORWARD`
- [ ] Step 3: 更新测试 include / dual-path 测试（改为仅 layer）

---

### Task 5: 验证

```bash
XMAKE_ROOT=y xmake f --nv-gpu=y && XMAKE_ROOT=y xmake && XMAKE_ROOT=y xmake install
PYTHONPATH=./python python test/test_infer.py --device nvidia --model ... --test --max_step 8
PYTHONPATH=./python python test/test_llama_smoke.py --device nvidia --max-steps 4
```

- [ ] Step 1: 构建通过
- [ ] Step 2: Qwen2 + Llama 短生成 PASS
- [ ] Step 3: Commit `refactor: align Model base and drop qwen2 legacy forward`
