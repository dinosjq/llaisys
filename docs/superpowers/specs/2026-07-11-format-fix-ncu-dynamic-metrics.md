# Benchmark Format Fix & NCU Dynamic Metrics — Design Spec (v2)

Date: 2026-07-11

## Context

`--use-ncu` 重构完成后，验证发现三个问题：
1. 表格输出前有 18 行空白（循环里 `print(format_summary(""))`）
2. 行编号带 `#` 前缀，应改为纯数字
3. NCU 指标采集失败：`dram__throughput` 和 `dram__bytes` 在 RTX 4060（Ada Lovelace）上不存在，硬编码指标名不可靠

## Fix 1: 去掉表格前的空白行

**修复**：删除所有 12 个 benchmark 脚本循环里的 `print(format_summary(result))`。最终表格仅由 `save_results()` 输出。

## Fix 2: 去掉行编号的 "#" 前缀

**修复**：`#0 ` → ` 0 `

## Fix 3: NCU 动态指标发现

**根因**：硬编码的指标名在不同 GPU 架构上不可移植。RTX 4060 (Ada Lovelace) 经过实验验证：

| 我的硬编码 | RTX 4060 实际 |
|---|---|
| `gpu__time_duration.sum` | ✅ |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ✅ |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | ❌ 应为 `_sustained_active` |
| `dram__throughput.*` | ❌ 不存在 |
| `dram__bytes_*` | ❌ 不存在 |
| `dram__cycles_elapsed.*` | ✅ |

**修复策略**：

1. 用 `ncu --list-metrics` 查询本机可用指标名，缓存到 `results/.ncu_metrics_cache`（带 GPU 型号 key，24h TTL）
2. **前缀匹配**：`m.startswith(prefix)` 选择指标，`prefix` 以 `.` 结尾表示"取该前缀下的所有指标"
3. 期望前缀：

```python
DESIRED_PREFIXES = [
    "gpu__time_duration.sum",       # 精确，不以 . 结尾
    "sm__throughput.",               # 前缀：取所有后缀变体
    "sm__warps_active.",             # 前缀：取所有后缀变体（_elapsed 或 _active）
    "dram__cycles_elapsed.",         # 前缀：取所有后缀变体
    "dram__cycles_active.",          # 前缀：取所有后缀变体
]
```

4. `classify_bottleneck()` **不硬编码指标名**——从传入的 `kernel_data` dict 的 keys 中动态查找匹配的指标列。

## Fix 4: --use-ncu 不丢失表格 + 子进程正确限制 shape

**当前问题**：
- `if use_ncu: profile+return` → 表格没了
- 非 linear 脚本的子进程忽略 `--example-index`，跑全部 shape

**修复**：

1. `--use-ncu` 分支移到 `save_results()` **之后**
2. 所有 12 个脚本的 main loop 支持 `--example-index`：如果指定，只跑那些 shape
3. `_shape_args_for_index` 对非 linear 脚本传递 `--example-index <idx>`，子进程正确限制
4. `profile_benchmark` 接收 `example_indices`（已经过 auto-select 或用户指定），不自己重新跑 benchmark
5. NCU 命令包含 `--launch-skip`（跳过 tensor 创建期的 memcpy kernel）
6. Benchmark 跑两次是 NCU 的固有设计（第一次出表格，第二次 NCU 采集硬件计数器），在 README 中说明

**auto-select 逻辑**：在 benchmark 脚本的 `if use_ncu:` 分支中。如果用户未指定 `--example-index`，从 `results` 中自动选择：
- speedup < 1.0 → 选最小 speedup（差距最大）
- 全部 >= 1.0 → 选最接近 1.0 的

## Files

- `benchmark_harness.py`: 行编号 "#" → 纯数字
- 12 个 benchmark 脚本: 删除 `print(format_summary(result))`，main loop 支持 `--example-index`
- `ncu_profiler.py`: 添加动态指标发现（前缀匹配 + 缓存 + 动态 classify_bottleneck）

## Verification

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --M 128 --variant q_proj
# 预期：无空白行，行编号为纯数字

python test/benchmark/ops/benchmark_linear.py --dtype f16 --use-ncu --example-index "3" --warmup 5 --repeat 5
# 预期：先输出表格，然后 NCU profile（指标非零）
```
