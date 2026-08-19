# 推理优化文档索引

按**任务类型**汇总；Flash Decoding 原文另存 archive 便于核对。

| 文档 | 内容 |
|---|---|
| [01-baseline-and-correctness.md](./01-baseline-and-correctness.md) | 环境、正确性/初始化/P0·D0·D1/nsys 基线；prefix·sampling·request 修复 |
| [02-flash-decoding.md](./02-flash-decoding.md) | Flash Decoding 通读重写汇总（方法/数据/废弃路径/ vs FlashInfer） |
| [03-model-layer-framework.md](./03-model-layer-framework.md) | 模型 / Layer / Context 分层框架（含原 §3/§4 时间线） |
| [04-linear-gemv.md](./04-linear-gemv.md) | Linear 小 batch GEMV：M=1–15 重构、Qwen2/Llama 扫表、失败路径与限制 |
| [05-inference-baseline.md](./05-inference-baseline.md) | **当前端到端 + nsys 正式基线**（配置、吞吐、P0/D0/D1） |
| [06-phase4-decode-prep.md](./06-phase4-decode-prep.md) | 下一阶段准备：短 decode / Llama gate·down / 验收口径 |
| [07-w8a16-performance.md](./07-w8a16-performance.md) | **W8A16**：linear microbench + Qwen e2e FP vs INT8、显存与结论 |
| [archive/flash-decoding/](./archive/flash-decoding/) | 各版原始专题报告（v1–v9、演进、ncu 等） |

**相关收口报告（不在本目录）：**
[docs/reports/2026-08-16-phase3-closeout.md](../reports/2026-08-16-phase3-closeout.md)

**端到端基线（2026-08-17）：** 对称 HF `generate` 对照；主 KPI 为 vs HF-serial。详见
[05-inference-baseline.md](./05-inference-baseline.md)。

**GEMV：** 当前实现覆盖 `M=1..15`；详见 [04-linear-gemv.md](./04-linear-gemv.md)。

**W8A16（2026-08-19）：** 朴素 `linear_w8a16` vs BF16 GEMV；Qwen 1.5B e2e 吞吐约持平、清加载副本后显存约 −29%。详见
[07-w8a16-performance.md](./07-w8a16-performance.md)。

---

## 历史阶段文档对照

原先按日期命名的「阶段交付」已在 commit `10c737b`（`docs: consolidate optimization reports by task type`）合并进上表；**文件仍在 git 历史中**，工作区改为任务型汇总。

| 原路径（已从工作区移除） | 现位置 |
|---|---|
| `2026-08-01-00-baseline.md` | → [01-baseline-and-correctness.md](./01-baseline-and-correctness.md)（更早合并） |
| `2026-08-01-01-correctness-foundation.md` | → 同上 |
| Flash Decoding 各版 `2026-08-12-…` / `2026-08-13-…` | 汇总 → [02-flash-decoding.md](./02-flash-decoding.md)；原文 → [archive/flash-decoding/](./archive/flash-decoding/) |
| `2026-08-15-02-model-layer-framework.md`（§3 骨架） | → [03-model-layer-framework.md](./03-model-layer-framework.md) §1 |
| `2026-08-15-03-qwen2-framework-migration.md`（§4 Qwen2） | → [03](./03-model-layer-framework.md) §2 |
| `2026-08-15-04-llama-vertical-slice.md`（§4 Llama） | → [03](./03-model-layer-framework.md) §3 |
| `2026-08-16-05-unified-model-api.md` | → [03](./03-model-layer-framework.md) §4 |

查看合并前原文示例：

```bash
git show 10c737b^:docs/optimization/2026-08-15-02-model-layer-framework.md
git show 10c737b^:docs/optimization/2026-08-15-03-qwen2-framework-migration.md
git show 10c737b^:docs/optimization/2026-08-15-04-llama-vertical-slice.md
git show 10c737b^:docs/optimization/2026-08-16-05-unified-model-api.md
```

计划类（未删）：`docs/superpowers/plans/2026-08-15-*.md`、`2026-08-16-phase3-4-closeout.md`。
