# Qwen2 接入 Model / Layer 框架 — 交付报告

**日期:** 2026-08-16  
**分支:** `feat/model-layer-framework`  
**工作树:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**模型:** `/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B`

## 目标

将生产 Qwen2 从硬编码 `forward` 迁到语义 Layer 组装（Embedding / Attn / FFN），默认走 Layer；保留 legacy 回滚；P0/D0/D1 性能相对 legacy ≤3%。

## 迁移 commits（摘要）

| Task | Commit | 说明 |
|---|---|---|
| 1 | `70898c4` | `prepare_*` 上提到 `Model` |
| 2 | `1ee414f` | 双路径（默认 legacy） |
| 3 | `f4edc23` | greedy 硬门禁（42 tokens equal） |
| 4 | `933e3b5` | Python → Context Scheme A sync |
| 5 | `940ff20` | 默认 Layer |
| 6 | （本报告） | P0/D0/D1 性能门禁 |

## 硬门禁（Task 3）

- `test/test_qwen2_layer_parity.py`：legacy vs layer greedy token 完全一致（42 tokens）
- 生产路径 `stable_capture` 默认关闭（避免 CUDA teardown SEGV）；对照测试可开

## 性能门禁（Task 6）— `gate_status: passed`

脚本：`test/profile/migration_perf_gate.py`  
原始数据：`test/results/migration_perf_gate_20260816_020601.json`  
指标：warmup 后 median `wall_per_output_token`（ms/tok），3 次重复

| Shape | legacy median | layer median | Δ | ≤3% |
|---|---:|---:|---:|---|
| P0 (256→1) | 47.364 | 47.234 | -0.27% | PASS |
| D0 (1→128) | 16.381 | 16.041 | -2.08% | PASS |
| D1 (1024→128) | 18.302 | 18.487 | +1.01% | PASS |

**结论：** 保持默认 Layer；不回退。

## 回滚

```bash
LLAISYS_QWEN2_LAYER_FORWARD=0   # 强制 legacy
# unset / 其它值 → Layer（默认）
```

## 状态更新（2026-08-16）

- `forward_legacy` 与 `LLAISYS_QWEN2_LAYER_FORWARD` **已删除**；上表回滚方式失效。
- Scheme A（填 `LlaisysQwen2Weights` 再 sync）**已删除**；改为 `llaisysModelSetWeight(role, layer, tensor)`。
- 层组装在 `Qwen2::forward` 内完成，不再调用 `run_qwen2_layer_stack`。
- 终态说明：`docs/optimization/2026-08-16-05-unified-model-api.md`。

## 复现

```bash
cd /home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework
source ~/llaisys-main/venv/bin/activate
export PYTHONPATH=./python:$PYTHONPATH
XMAKE_ROOT=y xmake f --nv-gpu=y && XMAKE_ROOT=y xmake && XMAKE_ROOT=y xmake install

python test/profile/migration_perf_gate.py \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --shapes P0,D0,D1 --repeats 3 \
  --out test/results/migration_perf_gate_$(date +%Y%m%d_%H%M%S).json
```
