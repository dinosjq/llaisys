# §3/§4 交付收口 Implementation Plan

> **For agentic workers:** Execute task-by-task. Checkbox tracking.

**Goal:** 把「优化总计划 §3 Model/Layer 框架 + §4 迁移/Llama」相对**当前代码**仍缺的交付与记账补齐，并提交未落地的统一 API / `core` 命名空间改动。

**Architecture:** 代码主线已越过原 §3「共存 A」进入生产 Layer + 统一 C API；本计划只做文档对齐、小清理与 commit，不新开功能（无 rope_scaling / 无量化）。

**Tech Stack:** 现有 LLAISYS；交付写入 `docs/optimization/`。

**Context:**
- §3 原交付：`docs/optimization/2026-08-15-02-model-layer-framework.md`（骨架期，路径已过时）
- §4 原交付：`...-03-qwen2-framework-migration.md`、`...-04-llama-vertical-slice.md`（仍写 env 切栈 / Scheme A / `run_*_stack`）
- 工作区未提交：统一 `llaisysModel*`、`Llama : Qwen2`、组装在 `Model::forward`、`llaisys::core`、删旧 Qwen2 ABI

## Global Constraints

- 不改 Scheduler / Sequence / BlockManager
- 不实现 Llama-3 `rope_scaling`
- 不删用户中文注释；`sync_weights` 若改名须同步注释
- 不提交 `test/results/*.nsys-rep` 等大产物、`_task3_run_gpu_probe.sh`（除非明确需要）
- 用户已要求补全并执行；允许创建 commits

---

### Task 1: 写清「缺什么」的收口计划（本文件）

- [x] **Step 1:** 保存本计划到 `docs/superpowers/plans/2026-08-16-phase3-4-closeout.md`

---

### Task 2: 代码小清理

**Files:**
- Modify: `src/models/core/model.hpp`、`model.cpp`
- Modify: `src/models/qwen2/qwen2.hpp`、`qwen2.cpp`

- [ ] **Step 1:** 将虚函数 `sync_weights()` 改名为 `bind_kv_caches()`（仅绑 `_k_cache`/`_v_cache` → `_ctx`）；更新 `Model::start` 与 `Qwen2`/`Llama`/`forward` 内调用与中文注释。
- [ ] **Step 2:** `xmake` 编译通过。

---

### Task 3: 更新 / 追加交付文档

**Files:**
- Modify: `docs/optimization/2026-08-15-02-model-layer-framework.md` — 文末加「后续演进」指向当前布局（不改写历史验收表）
- Modify: `docs/optimization/2026-08-15-03-qwen2-framework-migration.md` — 文末加「后继：统一 API / 删除 legacy」
- Modify: `docs/optimization/2026-08-15-04-llama-vertical-slice.md` — 文末加「后继：`Llama : Qwen2` + Create(LLAMA)」
- Create: `docs/optimization/2026-08-16-05-unified-model-api.md` — **本轮主交付**：目录、`llaisys::core`、C API、删除项、验证命令与结果

- [ ] **Step 1:** 写 `05` 交付报告（含验证表：symbols / infer / llama smoke / layer cpu）。
- [ ] **Step 2:** 给 02/03/04 各加一小节「状态更新（2026-08-16）」。
- [ ] **Step 3:** 总计划文件若可写：将 `llama-vertical-slice` 标完成说明（`.cursor/plans/...` 用户侧；仓库内可用 optimization 索引一句带过）。

---

### Task 4: 验证

- [ ] **Step 1:** `XMAKE_ROOT=y xmake && xmake install`
- [ ] **Step 2:** `PYTHONPATH=python` 跑 `test_model_api_symbols`、`test_qwen2_request_api`、`test_qwen2_weight_map`
- [ ] **Step 3:** 若 GPU 可用：短 `test_infer`（`--max_step 4`）+ `test_llama_smoke --max-steps 4`
- [ ] **Step 4:** `xmake run llaisys-model-layer-cpu-test`

---

### Task 5: Commits

建议两个 commit（清晰记账）：

1. `refactor: unify model C API and rename framework namespace to core`  
   （代码 + Python + 测试 + CLAUDE.md；不含 results）
2. `docs: close out model-layer phase 3/4 delivery after unified API`  
   （optimization 交付 + 本计划 + specs 状态）

- [ ] **Step 1:** `git add` 相关路径（排除 `test/results/`、`_task3_run_gpu_probe.sh`）
- [ ] **Step 2:** 按上序提交
- [ ] **Step 3:** `git status` 确认干净（或仅剩故意未跟踪产物）

---

## Spec coverage

| 缺口 | Task |
|---|---|
| §3 文档路径过时 | 3（02 附录） |
| §4 仍写 env/Scheme A/stack | 3（03/04 附录 + 05） |
| 统一 API 无交付报告 | 3（05） |
| `sync_weights` 名实不符 | 2 |
| 未提交代码 | 5 |
| 冒烟确认 | 4 |
