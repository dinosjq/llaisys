# Model / Layer 框架（并行骨架）

日期：2026-08-15  
分支：`feat/model-layer-framework` @ `5e24a87`（交付文档 commit 另计）  
规格：`docs/superpowers/specs/2026-08-15-model-layer-framework-design.md`

## 目标

并行落地语义级 Model / Layer 框架，使后续模型可复用调度与 KV，并按 Embedding / Attn / FFN 组装 forward。

本阶段**不**切换生产 `Qwen2` 硬编码路径、**不**改 Python API、**不做** P0/D0/D1 性能对比门禁（归第 4 部分）。

## 基线

| 项 | 值 |
|---|---|
| 回退基线 | `main@86693ca`（上一版按算子拆 Layer 的 Pipeline 已回退） |
| 备份 | `backup/model-pipeline-before-revert`（`c77a839`） |
| 生产路径 | `src/models/qwen2.cpp` / `qwen2.hpp` / `python/` 相对 `86693ca` **无改动** |
| 共存策略 | A：旧硬编码不动；新框架仅样例 + 测试 |

## 设计（C+D + 文件组织）

### C 运行面 + D 上传面

| 面 | 内容 | 热路径 |
|---|---|---|
| **C** | `ModelContext`：`meta` / `runtime` / `workspace` + `AttnWeights[n]` / `FfnWeights[n]` + 全局 embed/norm | Layer 构造时绑定指针；`forward` 只用 `tensor_t` |
| **D** | 公共 `WeightRole` + `layer_idx` → `Model::set_weight` | 仅加载/测试注入；热路径无 string/role 查找 |

### 文件组织（相对早期草案的定稿）

| 约定 | 说明 |
|---|---|
| `weight_set` 并入 `Model` | 删除独立 `weight_set.*`；`set_weight` 与 `prepare_*` TODO 留在 `model.hpp` / `model.cpp` |
| `Layer` 仅头文件 | `framework/layer.hpp` 为纯虚接口，无 `.cpp` |
| Embedding / Attn / FFN 分离 | `layers/qwen2/{embedding,attention,ffn}.{hpp,cpp}` 各自实现语义块 |
| 显式组装 | `run_qwen2_layer_stack`：embedding → 各层 attn+ffn → last-token → out-norm → lm-head |
| 捕获点 | 层 0 attn/ffn **残差 swap 之后**，用 `stable_capture`（`tensor->to(CPU)`）做独立 CPU 快照 |
| Decode | NVIDIA 测试先 prefill 填满同上下文/KV，再 `is_prefill=false` 跑 decode；拒绝 NaN |

```text
src/models/framework/
  layer.hpp          # Layer 基类（header-only）
  model.hpp/.cpp     # Model + set_weight（原 weight_set）
  model_context.hpp  # C 分层权重 + workspace（含 capture_*）
  weight_role.hpp    # D 上传角色枚举

src/models/layers/qwen2/
  embedding.*  attention.*  ffn.*  stack.*
```

## 修改文件

相对 `86693ca` 新增/修改（生产 Qwen2 / Python 除外）：

- `src/models/framework/*`
- `src/models/layers/qwen2/*`
- `test/models/qwen2_layer_cpu_test.cpp`
- `test/models/qwen2_layer_nvidia_test.cpp`
- `test/core/core_test.cpp`（`set_weight` 槽位烟测）
- `xmake.lua`（`llaisys-model-layer-cpu-test` / `llaisys-model-layer-nvidia-test`）
- 附带：`src/ops/add/op.cpp` 小改（样例路径构建期连带）

**未改**：`src/models/qwen2.cpp`、`src/models/qwen2.hpp`、`python/`。

## 正确性

| 测试 | 结果 |
|---|---|
| `xmake build/run llaisys-core-test` | PASS（含预期的空 prompt Sequence 拒绝日志） |
| `llaisys-model-layer-cpu-test` | PASS：`qwen2 embedding and ffn CPU layer tests passed` |
| `llaisys-model-layer-nvidia-test` | PASS：prefill 层 0 PostAttn/PostFfn 与 legacy 对照；decode（prefill 后）对照；CPU 快照不与 workspace alias |
| `git diff --name-only 86693ca..HEAD -- src/models/qwen2.cpp src/models/qwen2.hpp python/` | 空 |
| 端到端 `test_infer.py` | 本交付环境无 `nvidia-smi`，跳过；不因缺模型/GPU 判失败 |

捕获约定：`capture_post_attn0` / `capture_post_ffn0` 为残差后的 CPU 副本，与可复用的 `x` / `x_attn` / `x_mlp` 存储分离。

## 已知限制

- 生产仍走硬编码 `Qwen2::forward`；样例栈不接 worker。
- `prepare_*` 仅 TODO 占位（YAGNI，避免碰生产）。
- Python 仍用现有 `llaisysQwen2*`；角色上传未接生产加载器。
- 切换默认路径、P0/D0/D1、Llama 均归第 4 部分。
- 本环境 `clang-format` 不可用；xmake 仍有既有 `add_cuflags` 告警。

## 回滚

删除 `src/models/framework/`、`src/models/layers/`、相关 `test/models/qwen2_layer_*`、`xmake.lua` 测试 target 与本文档即可。生产推理不受影响。

## commits

| Hash | Message |
|---|---|
| `a038de5` | feat: add model context and weight role upload |
| `12223de` | feat: add qwen2 embedding and ffn sample layers |
| `e96f813` | feat: add qwen2 attention sample and nvidia capture check |
| `d255fde` | fix: preserve layer attention decode state |
| `bdc536b` | fix: stabilize model-layer captures and restructure Model weights |
| `5e24a87` | docs: record task-3 follow-up commit hash in report |
| （本交付） | docs: deliver model layer framework phase report |
