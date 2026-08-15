# Qwen2 接入框架并替换硬编码 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把第 3 部分并行框架接到生产 `Qwen2`：先上提 `prepare_*`、双路径对照验证通过，再默认切换到 Layer 组装并去掉硬编码 `forward`；Python 用 `WeightRole` 映射（兼容壳）；最后用同一框架接 Llama 纵向切片。

**Architecture:** `Qwen2` 继承/持有 `framework::Model`；worker 仍用现有 scheduler/KV/request API。运行期 `ModelContext`（C）+ 已有 Embedding/Attn/FFN；加载期 `WeightRole`（D）经 `Model::set_weight`。迁移期保留硬编码 `forward_legacy` 作对照与回滚，默认切换后门禁通过后再删除或 `#if` 冻结。

**Tech Stack:** C++17、现有 ops、ctypes/Python safetensors、xmake、NVIDIA P0/D0/D1（`docs/常用命令.md`）、`huggingface-cli`（缺权重时经 CN 镜像下载）。

## Global Constraints

- 基线分支：先合并或基于 `feat/model-layer-framework`（含第 3 部分骨架）；不恢复已回退的按算子拆 Layer Pipeline。
- **先验证、后切换**：未通过「硬编码 vs Layer」捕获点 + greedy 硬门禁前，生产默认必须仍是硬编码路径。
- 硬门禁：同种子 greedy token 完全一致；`PostAttn[0]` / `PostFfn[0]` / `FinalLogits`（或 PostBlock）CPU `stable_capture` 一致（dtype 容差按 F32/BF16 约定）。
- HF 逐 token：仅诊断，不作硬门禁。
- 切换后 P0/D0/D1：median TTFT/TPOT（或 `wall_per_output_token`）相对切换前同环境回退 ≤ **3%**；不通过则默认退回 legacy 并保留报告。
- Python：`generate` / `generate_async` / server 对外行为不变；内部可改为角色上传或「先填旧 Weights 再 sync 到 Context」。
- 代码风格：对齐现有 `qwen2.cpp`（中文步骤注释、局部变量别名）；Emb/Attn/FFN 可继续分文件；`set_weight` 留在 `Model`，不恢复独立 `weight_set.*`。
- Layer = 语义块；禁止按算子拆 Layer。
- Llama 不改调度/请求/KV 生命周期；仅自实现 Layer + 映射表。
- 模型目录（与现网一致的父路径 `/home/songjq/models/`）：
  - Qwen：`/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B`
  - Llama：`/home/songjq/models/Llama-3.2-3B-Instruct`
- **缺权重时必须下载，不得用假权重冒充真实模型门禁**。优先 CN 镜像：
  ```bash
  export HF_ENDPOINT=https://hf-mirror.com
  huggingface-cli download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local-dir /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B
  huggingface-cli download meta-llama/Llama-3.2-3B-Instruct \
    --local-dir /home/songjq/models/Llama-3.2-3B-Instruct
  ```
  下载可能较慢，允许后台拉取；Llama 若需授权先 `huggingface-cli login`。
- 不推送远端；每 Task 独立 commit；中文交付文档 + 更新 Canvas。
- TDD；不得靠放宽误差换通过。

---

## File Structure（预期落点）

| Path | Responsibility |
|---|---|
| `src/models/framework/model.hpp/.cpp` | `set_weight`；上提后的 `prepare_block_table` / `prepare_prefill` / `prepare_decode`（或共享 free functions 由 Model 调用） |
| `src/models/framework/model_context.hpp` | 已有 C 面；按需补采样字段、`tot_block_num`、与 `Qwen2Pack` 互转 |
| `src/models/framework/pack_bridge.hpp`（可选） | `Qwen2Pack` ↔ `BatchRuntime` + workspace H2D |
| `src/models/layers/qwen2/*` | 已有语义层；补齐与生产一致的采样前 logits 路径 |
| `src/models/qwen2.hpp/.cpp` | 持有 `ModelContext` + layers；`forward_legacy` / `forward_layers`；worker 调用可切换入口 |
| `include/llaisys/models/qwen2.h` + `src/llaisys/models/qwen2.cc` | 可选：`set_weight(role,layer,tensor)`；或保持旧 Weights 填充再 C++ sync |
| `python/llaisys/models/qwen2.py` + 小映射表 | HF 名 → `(WeightRole, layer)`；对外 API 不变 |
| `test/models/qwen2_migration_*.cpp` / Python 对照 | 双路径捕获 + greedy |
| `docs/optimization/2026-08-15-03-qwen2-framework-migration.md` | 交付 |
| 稍后 Llama | `src/models/layers/llama/*` + profile 映射；交付 `...-04-llama-vertical-slice.md` |

---

### Task 1: 上提 `prepare_*` 到 Model（行为不变）

**Files:**
- Modify: `src/models/framework/model.hpp`, `model.cpp`
- Modify: `src/models/qwen2.cpp` / `qwen2.hpp`（改为调用基类/共享实现）
- Modify: `test/core/core_test.cpp` 或新建 `test/models/prepare_parity_test.cpp`

**Interfaces:**
- Produces（签名与现 `Qwen2` 对齐，可放在 `llaisys::framework` 或 `Model` static/成员）:
  - `std::vector<int64_t> prepare_block_table(const std::vector<seq_t> &seqs, size_t block_table_width);`
  - `BatchRuntime` 或继续返回 `Qwen2Pack` 的 `prepare_prefill` / `prepare_decode`（**本 Task 优先保持 `Qwen2Pack` 返回类型**，避免一次改太多；下一 Task 再桥到 Context）
- Consumes: 现有 `Sequence` / `Qwen2Pack`

- [ ] **Step 1: 写失败的行为对照测试**

对固定 2 条 seq（不同 cached/scheduled），断言新 `Model::prepare_*`（或 free function）与**当前** `Qwen2::prepare_*` 输出向量逐元素相等。先把旧实现拷到测试作 oracle，再搬代码——或：提取前先测 Qwen2，提取后同一输入仍相等。

- [ ] **Step 2: RED** — 若先加空声明则链接/逻辑失败；按 TDD 先锁定期望向量。

- [ ] **Step 3: 把 `qwen2.cpp` 中三函数实现移到 `model.cpp`（保留中文注释），`Qwen2::prepare_*` 一行转发。**

- [ ] **Step 4: 跑 core / prepare 测试 + 确认 `qwen2` 生产路径仍默认硬编码 forward。**

```bash
xmake build llaisys-core-test && xmake run llaisys-core-test
```

- [ ] **Step 5: Commit** `refactor: lift qwen2 prepare helpers into Model`

---

### Task 2: Qwen2 持有 Context + Layers，双路径 forward（默认仍 legacy）

**Files:**
- Modify: `src/models/qwen2.hpp`, `qwen2.cpp`
- Modify: `src/models/layers/qwen2/stack.*`（扩展为可产出 logits，接采样前逻辑；或 Qwen2 在 stack 后自行 topk/sample）
- Modify: `xmake.lua` if needed
- Create: `test/models/qwen2_dual_path_nvidia_test.cpp`（或扩展现有 nvidia layer test）

**Interfaces:**
- `Qwen2` 增加：`framework::ModelContext _ctx`（或等价）、`std::vector` 层对象 / 每次栈上构造 Attn/FFN（与第 3 部分一致即可）
- `forward_legacy(pack, block_ids, is_prefill)` = 现有循环（可改名）
- `forward_layers(...)` = 填 Context ← pack/block_ids/KV，再 `run_qwen2_layer_stack`，再 **与 legacy 相同的** topk + `sample_token` 后处理
- 运行时开关：环境变量或成员 `bool use_layer_forward_{false}`；**默认 false**
- Worker 循环：`use_layer_forward_ ? forward_layers : forward_legacy`

- [ ] **Step 1: 失败测试** — 打开 layer 路径时，tiny 权重下 `forward_layers` 与 `forward_legacy` 的 logits/greedy 不一致则先红（实现前）。

- [ ] **Step 2: 实现 Context 填充**：从现有 `_weights` / `_tensors` / `_k_cache` 绑定到 `_ctx`（构造或 finalize 时一次 bind；pack 每步写入 runtime + H2D workspace）。

- [ ] **Step 3: `forward_layers` + 默认仍走 legacy。**

- [ ] **Step 4: NVIDIA 双路径测试 PASS（开关显式 true 时）。**

- [ ] **Step 5: Commit** `feat: add qwen2 dual forward path behind flag`

---

### Task 3: 完整对照门禁（真实模型短生成）

**Files:**
- Create: `test/test_qwen2_layer_parity.py`（或扩展现有 infer test）
- Modify: 测试文档 / 交付草稿

**Hard gate:**
1. NVIDIA tiny：已有捕获点 + decode-after-prefill
2. 1.5B BF16：`use_layer_forward=true` vs `false`，同 prompt、temperature≤0（greedy），`max_step`≥32，**token 序列完全一致**
3. 可选：首末层 capture（若启用 test build）一致

```bash
# 示例（实现时写死到脚本）
LLAISYS_QWEN2_LAYER_FORWARD=0 python test/test_infer.py --device nvidia --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --test --max_step 32
LLAISYS_QWEN2_LAYER_FORWARD=1 python test/test_infer.py ...  # 断言与上一跑 token 一致，或单进程内双模型对照
```

HF 对比只打印差异，不 fail。

- [ ] **Step 1: 写对照脚本（失败当不一致）。**
- [ ] **Step 2: 修到 PASS。**
- [ ] **Step 3: Commit** `test: enforce qwen2 legacy vs layer greedy parity`

---

### Task 4: Python WeightRole 映射（兼容壳）

**Files:**
- Create: `python/llaisys/models/weight_roles.py`（或 `qwen2_weight_map.py`）：`HF_NAME → (role, layer)`
- Modify: `python/llaisys/models/qwen2.py` `_assign_weight`
- Modify: C ABI — **优选最小改动**：
  - **方案 A（推荐）**：Python 仍写 `LlaisysQwen2Weights`；C++ 在权重加载结束 / `start` 前 `sync_weights_from_legacy_struct()` 填入 `ModelContext`
  - **方案 B**：新增 `llaisysQwen2SetWeight(role, layer, tensor)`，Python 直接 set_weight

本 Task 采用 **方案 A**，避免一次改 ctypes 面；方案 B 留作可选后续。

- [ ] **Step 1: 单测映射表覆盖 DeepSeek-Qwen2 关键键（含 bias / tie）。**
- [ ] **Step 2: sync 后 layer 路径 greedy 与 Task 3 一致。**
- [ ] **Step 3: Commit** `feat: sync qwen2 weights into model context via roles`

---

### Task 5: 默认切换到 Layer 路径 + 冻结 legacy

**Files:**
- Modify: `qwen2.cpp` 默认 `use_layer_forward_=true`（或编译宏 `LLAISYS_QWEN2_LAYER_DEFAULT=1`）
- 保留 `forward_legacy` 经环境变量 `LLAISYS_QWEN2_LAYER_FORWARD=0` 回滚
- Modify: server / 文档说明

**Gate before flipping default:** Task 3 PASS 记录在案。

**Rollback:** `LLAISYS_QWEN2_LAYER_FORWARD=0` forces legacy `forward_legacy` (unset/other → layer).

- [x] **Step 1: 切换默认；跑 `test_infer` + `test_server`（端口自选）+ layer/cpu/nvidia tests。**
- [x] **Step 2: Commit** `feat: default qwen2 inference to layer stack`

---

### Task 6: P0/D0/D1 性能门禁（≤3%）

**Files:**
- Create/Modify: `test/profile/` 或复用 `nsys_profile.py` 对照脚本
- Create: `docs/optimization/2026-08-15-03-qwen2-framework-migration.md`（本部分交付可在此 Task 完成）

**Protocol:**
1. 同 commit 树、同机器：legacy 默认关 vs layer 默认开，各跑 P0/D0/D1（warmup 后 median）
2. 指标：`wall_per_output_token` 或 TPOT；TTFT 一并记录
3. 回退 >3% → **默认改回 legacy**，文档 `gate_status: failed`，不删 layer 代码

```bash
# 按 docs/常用命令.md；保存 test/results/migration_* 
```

- [x] **Step 1: 采 baseline（legacy）与 layer 各至少 3 次 median。**
- [x] **Step 2: 写交付文档（目标/对照数据/commits/回滚）。**
- [x] **Step 3: 更新 Canvas 第 4 部分状态。**
- [x] **Step 4: Commit** `docs: report qwen2 framework migration gates`

---

### Task 7: Llama-3.2 纵向切片（同框架，不改调度）

**Files:**
- Create: `src/models/layers/llama/{attention,ffn,embedding,stack}.*`（可先复用 Qwen2 FFN/Attn 若结构一致，差在 bias/norm）
- Create: Python `llama` 映射或通用 loader + profile
- Test: tiny synthetic + 短生成；OOM 则 CPU/缩 KV

**Acceptance:** 不改 `Scheduler`/`Sequence`/`BlockManager`；第三模型仅 Layer + 映射。

- [x] **Step 1–N:** 按 TDD 接 Llama；交付 `docs/optimization/2026-08-15-04-llama-vertical-slice.md`
- [x] **Commit** `feat: run llama through model layer framework`

---

## 建议执行顺序与回滚

```text
Task1 prepare 上提
  → Task2 双路径（默认 legacy）
  → Task3 硬门禁对照
  → Task4 Python sync
  → Task5 切换默认
  → Task6 性能门禁（失败则默认回退）
  → Task7 Llama
```

回滚：`LLAISYS_QWEN2_LAYER_FORWARD=0` 或恢复默认 legacy；framework 文件可保留。

---

## Spec coverage

| 规格 9.2 / 用户要求 | Task |
|---|---|
| prepare 公共部分进 Model | 1 |
| 接入并测试框架 | 2–3 |
| 验证通过后替换硬编码 | 5（3 为前置门禁） |
| Python WeightRole / 兼容 | 4 |
| P0/D0/D1 ≤3% | 6 |
| Llama 同框架 | 7 |
| 策略：先验证后切换 | Global + Task 2 默认 false |

## 明确不做（本计划内）

- 量化、CUDA Graph、投影融合
- 按算子拆 Layer
- 恢复独立 `weight_set` 文件
- 未过 Task 3 就切换默认
