---
name: perf-baseline-workflow
description: 每次优化前后必须记录完整性能数据并保存到特定文件夹做对比
metadata:
  type: project
---

任何算子/系统优化开始前，必须先跑完整性能数据并保存，优化后再跑一遍做对比。

**Why:** 没有 baseline 的优化无法量化收益。面试和文档都需要 A/B 对比数据（[[perf-benchmark-report]]）。同一个优化方向可能有多个方案，数据驱动选择最佳方案。

**How to apply:**

## 文件夹结构

性能数据统一保存在 `docs/perf/` 下，按优化主题分子文件夹：

```
docs/perf/
├── 2026-07-17_paged_attn_optimization/
│   ├── before/                          # 优化前 baseline
│   │   ├── benchmark_1024.json          # max_step=1024 的完整 benchmark
│   │   ├── benchmark_2048.json          # 可选的更长序列数据
│   │   ├── profile.json                 # profile_model.py 的算子级耗时
│   │   └── notes.md                     # 环境信息（GPU型号、驱动版本、编译选项等）
│   ├── after/                           # 优化后
│   │   ├── benchmark_1024.json
│   │   ├── profile.json
│   │   └── notes.md                     # 改动摘要
│   └── comparison.md                    # before vs after 对比分析
├── 2026-07-20_linear_fusion/
│   ├── before/
│   ├── after/
│   └── comparison.md
└── ...
```

## 需要采集的数据

| 数据项 | 采集方式 | 说明 |
|--------|----------|------|
| 完整 Benchmark | `python test/benchmark/benchmark_batch_infer.py --model <path>` | 多 max_step 挡位（128/256/512/1024/2048） |
| 算子级 Profile | `python test/profile/profile_model.py --model <path>` | per-operator 耗时分布 |
| 吞吐量 | benchmark 输出中的 tokens/sec | 需要 batch 并发测试 |
| GPU 显存 | `nvidia-smi` 或 CUDA API 查询 | 峰值显存、KV cache 占用 |
| 环境信息 | 手动记录 | GPU 型号、驱动版本、CUDA 版本、编译选项 |

## 对比分析 (comparison.md) 模板

```markdown
# <优化名称> 性能对比

## 环境
- GPU: RTX 4060 Laptop (8GB)
- 编译: xmake f --nv-gpu=y -cv && xmake && xmake install

## 改动摘要
- 文件: src/ops/paged_attention/nvidia/paged_attention_nvidia.cu
- 改动: <一句话描述>

## 算子级耗时对比

| 算子 | before (ms) | after (ms) | 变化 |
|------|-------------|------------|------|
| paged_attn | 17.98 | xxx | -xx% |
| linear | 27.0 | 27.0 | 0% |
| ... | | | |

## 端到端对比

| max_step | before (tokens/sec) | after (tokens/sec) | 加速比 |
|----------|---------------------|--------------------|--------|
| 128 | | | |
| 1024 | | | |

## 结论
```

## 执行命令参考

```bash
# 1. 优化前 baseline
mkdir -p docs/perf/YYYY-MM-DD_<topic>/before
python test/profile/profile_model.py --model <path> --max-step 1024 \
    --output docs/perf/YYYY-MM-DD_<topic>/before/profile.json
python test/benchmark/benchmark_batch_infer.py --model <path> \
    --output docs/perf/YYYY-MM-DD_<topic>/before/

# 2. (做优化...)

# 3. 优化后
mkdir -p docs/perf/YYYY-MM-DD_<topic>/after
python test/profile/profile_model.py --model <path> --max-step 1024 \
    --output docs/perf/YYYY-MM-DD_<topic>/after/profile.json
python test/benchmark/benchmark_batch_infer.py --model <path> \
    --output docs/perf/YYYY-MM-DD_<topic>/after/

# 4. 写 comparison.md
```
