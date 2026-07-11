# Benchmark Format Fix & NCU Dynamic Metrics — Design Spec

Date: 2026-07-11

## Context

`--use-ncu` 重构完成后，验证发现三个问题：
1. 表格输出前有 18 行空白（循环里 `print(format_summary(""))`）
2. 行编号带 `#` 前缀，应改为纯数字
3. NCU 指标采集失败：`dram__throughput` 和 `dram__bytes` 在 RTX 4060（Ada Lovelace）上不存在，硬编码指标名不可靠

## Fix 1: 去掉表格前的空白行

**根因**：每个 shape 的 benchmark 循环结束后调用 `print(format_summary(result))`，而 `format_summary` 返回 `""`。`print("")` 输出换行 → 18 行空白。

**修复**：删除所有 12 个 benchmark 脚本循环里的 `print(format_summary(result))`。最终表格仅由 `save_results()` 输出。

## Fix 2: 去掉行编号的 "#" 前缀

**当前**：`#0  128 x 3584 x 3584 x q_proj`

**修复**：` 0  128 x 3584 x 3584 x q_proj`

## Fix 3: NCU 动态指标发现

**根因**：硬编码的指标名 `dram__throughput.avg.pct_of_peak_sustained_elapsed`、`dram__bytes_read.sum`、`sm__warps_active.avg.pct_of_peak_sustained_elapsed` 在 RTX 4060 上不存在或名称不同。

**经实验验证（RTX 4060）**：
| 期望 | 实际存在 |
|---|---|
| `gpu__time_duration.sum` | ✅ |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | ✅ |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | ❌ 应为 `sm__warps_active.avg.pct_of_peak_sustained_active` |
| `dram__throughput.*` | ❌ 不存在 |
| `dram__bytes_*` | ❌ 不存在 |
| `dram__cycles_elapsed.*` | ✅ |

**修复**：`profile_benchmark()` 启动时调用 `ncu --list-metrics`，用期望指标名的**前缀**匹配实际可用指标。缓存到 `results/ncu/.metrics_cache` 避免每次调用。

**期望前缀列表**：
```python
DESIRED_PREFIXES = [
    "gpu__time_duration.sum",
    "sm__throughput.",
    "sm__warps_active.",
    "dram__cycles_elapsed.",
    "dram__cycles_active.",
]
```

## Fix 4: --use-ncu 不丢失表格

**当前**：`if use_ncu: profile+return` → 表格没了

**修复**：`--use-ncu` 先正常跑 benchmark → 输出表格 → 然后 profile：

```
for shape: run_benchmark()  → 收集 results
save_results(results)        → 输出表格 + JSON
if use_ncu: profile(results) → NCU → 瓶颈输出
```

## Files

- `benchmark_harness.py`: 行编号 "#" → 数字
- 12 个 benchmark 脚本: 删除 `print(format_summary(result))`
- `ncu_profiler.py`: 添加 `_get_available_metrics()` 动态指标发现, 使用前缀匹配
- `benchmark_linear.py`: `--use-ncu` 改为跑完表格后再 profile

## Verification

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --M 128 --variant q_proj
# 预期：无空白行，行编号为纯数字

python test/benchmark/ops/benchmark_linear.py --dtype f16 --use-ncu --example-index "3" --warmup 5 --repeat 5
# 预期：先输出表格，然后 NCU profile（指标非零）
```
