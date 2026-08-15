# 推理优化文档索引

本目录按**任务类型**合并专题报告。旧的按日期拆分文件已删除；内容收入下列汇总。

| 文档 | 内容 |
|---|---|
| [01-baseline-and-correctness.md](./01-baseline-and-correctness.md) | 环境、正确性/初始化/P0·D0·D1/nsys 基线；prefix·sampling·request 语义修复 |
| [02-flash-decoding.md](./02-flash-decoding.md) | Flash Decoding 演进主线 + 各版关键数据/方法/废弃实验 |
| [03-model-layer-framework.md](./03-model-layer-framework.md) | Model/Layer 骨架 → Qwen2 迁移门禁 → Llama → 统一 C API |

**刻意不收录：** GEMV 自研实验（相对 cublas 无收益；当时评测 batch≥16 落在 cublas Tensor Core 友好区间，结论不适用 decode 小 batch 主场景）。

后续新阶段请直接新增 `04-…md` 并在本表追加一行，避免再按日期碎片化。
