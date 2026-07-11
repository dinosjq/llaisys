# Operator Performance Benchmark & NCU Integration — Design Spec

Date: 2026-07-11

## Context

`src/ops/` 实现了 12 个 CUDA 算子，`test/ops/` 仅有正确性测试，缺乏性能测试。
`test/benchmark/` 目录已存在但为空。需要构建一套算子性能测试系统，集成 NCU (Nsight Compute) profiling 能力。

## Scope

- 创建 `test/benchmark/ops/` 目录，含共享 harness、NCU 模块、每个算子一个 benchmark 脚本
- 专注于 CUDA (NVIDIA) 算子，CPU 不在范围内
- `kv_cache_move` 和 `paged_attention` 不与基准对比（无直接等价实现）
- 基准：有 Triton 最优实现的用 Triton，否则用 PyTorch CUDA kernel

## File Structure

```
test/benchmark/ops/
├── __init__.py
├── benchmark_harness.py       # 共享测量框架 (CUDA event timing, L2 cache defeat, 统计)
├── ncu_profiler.py            # NCU 集成 (find_ncu, run_under_ncu, parse_csv, bottleneck classify)
├── benchmark_linear.py        # 每个算子一个脚本
├── benchmark_rms_norm.py
├── benchmark_rope.py
├── benchmark_swiglu.py
├── benchmark_embedding.py
├── benchmark_topk.py
├── benchmark_argmax.py
├── benchmark_add.py
├── benchmark_rearrange.py
├── benchmark_self_attention.py
├── benchmark_paged_attention.py    # 无基线对比
├── benchmark_kv_cache_move.py      # 无基线对比
├── run_all_benchmarks.py           # 批量入口
├── ncu_profile.sh                  # Shell 包装
└── results/                        # 输出目录
    ├── latest.json
    ├── YYYY-MM-DD/hhmmss.json
    └── ncu/*.ncu-rep
```

## Benchmark Harness (`benchmark_harness.py`)

### 核心类

```python
@dataclass
class BenchmarkConfig:
    op_name: str
    shapes: list[dict]        # [{"M": 512, "N": 3584, "K": 3584}, ...]
    dtypes: list[str]         # ["f16", "bf16"]
    warmup: int = 10
    repeat: int = 100
    device: str = "nvidia"

@dataclass
class BenchmarkResult:
    operator: str
    dtype: str
    shape: dict
    latency_us: float           # min across iterations (CUTLASS convention)
    latency_stats: dict         # {min, median, mean, stddev, p99}
    bandwidth_GBs: float
    TFLOPS: float               # 仅 compute-bound 算子有意义

def run_benchmark(config, setup_fn, llsys_fn, ref_fn=None) -> BenchmarkResult
```

### 测量协议

- **计时**：使用 CUDA Events（通过 RuntimeAPI 封装的 `cudaEventCreate/Record/ElapsedTime`），不再使用 `time.time()`
- **L2 cache 击败**：分配多组 buffer，每轮轮换，避免缓存残留导致虚高带宽
- **报告值**：以 min 作为主值（噪声只会增加延迟），同时输出 min/median/mean/stddev/p99
- **GPU 频率锁定**：benchmark 前打印提醒 `nvidia-smi -lgc <base_freq>`，未锁定时在报告中标记 `clock_locked: false`

### 基准映射

| 算子 | 基准实现 | 来源 |
|---|---|---|
| `linear` | cuBLAS (`torch.matmul`) | GEMM 天花板 |
| `rms_norm` | Unsloth Triton | H100 ~2911 GB/s (87% peak HBM) |
| `rope` | Unsloth fused QK-RoPE | 2.3x vs PyTorch, in-place |
| `swiglu` | Unsloth fused SwiGLU | 2.8x vs PyTorch |
| `self_attention` | Flash-Attention v2 Triton | 官方 tutorial, H100 ~180 TFLOPS |
| `top_k` | flexsae Triton | 1.7-4x vs torch.compile |
| `embedding` | PyTorch CUDA | 简单 gather，内存带宽天花板 |
| `add` | PyTorch CUDA | 简单 element-wise |
| `argmax` | PyTorch CUDA | 简单 reduction |
| `rearrange` | PyTorch CUDA (`contiguous()`) | 纯 memcpy |
| `kv_cache_move` | 无 | 自定义算子 |
| `paged_attention` | 无 | 自定义算子 |

### Shape 策略

单一档位，覆盖 decode/prefill 两端 + Qwen2 实际维度：

```
M (token 数): 1, 16, 64, 128, 512, 2048
(N, K) 组合取自 Qwen2 模型:
  - Q/K/V proj: (3584, 3584), (512, 3584)
  - O proj:     (3584, 3584)
  - Gate/Up:    (18944, 3584)
  - Down:       (3584, 18944)
总计每个算子 ~6 shapes × 2 dtypes = 12 个配置，可接受
```

## NCU 集成 (`ncu_profiler.py`)

### 双重启动模式（llm.c 风格）

```
用户运行: python benchmark_linear.py --ncu
  → benchmark_linear.py 构建 ncu 命令行
  → subprocess: ncu --set detailed --clock-control base ... python benchmark_linear.py --ncu-child
  → NCU 启动子进程，子进程只跑一次 kernel 然后退出
  → NCU 写入 .ncu-rep
  → 父进程用 ncu -i xxx.ncu-rep --csv 提取指标
  → 解析 → 瓶颈分类 → 输出报告
```

### NCUConfig

```python
@dataclass
class NCUConfig:
    set: str = "detailed"           # default | detailed | full
    clock_control: str = "base"
    launch_skip: int = 10           # 跳过 warmup
    launch_count: int = 1           # 只 profile 一次
    kernel_name: str = None         # regex 过滤
    replay_mode: str = "kernel"
    metrics: list = [...]           # 提取的指标列表
```

### 三档深度

| CLI flag | NCU --set | 开销 | 用途 |
|---|---|---|---|
| `--ncu quick` | `default` | ~3x | 瓶颈分类 |
| `--ncu detailed` | `detailed` | ~10x | 根因分析 |
| `--ncu full` | `full` | ~50x | GUI 深度分析 |

### 提取指标 (6 个核心指标)

```
gpu__time_duration.sum
dram__bytes_read.sum
dram__bytes_write.sum
sm__throughput.avg.pct_of_peak_sustained_elapsed
dram__throughput.avg.pct_of_peak_sustained_elapsed
sm__warps_active.avg.pct_of_peak_sustained_elapsed
```

### 瓶颈分类规则

```
dram_throughput > 60% AND sm_throughput < 20% → memory_bound
sm_throughput > 60% AND dram_throughput < 20% → compute_bound
both < 30% → launch_occupancy_bound
else → latency_bound
```

### CLI 接口

```bash
python benchmark_linear.py --M 512 --dtype f16              # 纯 benchmark
python benchmark_linear.py --ncu quick --M 512 --dtype f16  # NCU 快速
python benchmark_linear.py --ncu detailed --M 512            # NCU 深度
bash ncu_profile.sh --op linear,rope --set detailed           # Shell 包装
```

## 输出格式

### JSON (每次运行产出一个文件)

顶层字段: `environment`, `results[]`

每个 result:
```json
{
  "operator": "linear",
  "dtype": "f16",
  "shape": {"M": 512, "N": 3584, "K": 3584},
  "llaissys": {"latency_us": 142.3, "TFLOPS": 92.4, "bandwidth_GBs": 612.0, "latency_stats": {...}},
  "baseline": {"name": "cuBLAS", "latency_us": 105.2, "TFLOPS": 125.0, "speedup_vs_llaissys": 1.35},
  "ncu": {"set": "detailed", "report_file": "...", "bottleneck": "compute_bound", ...}
}
```

### 终端摘要

```
  linear  f16  M=512  N=3584  K=3584
  ─────────────────────────────────────────
    LLAISYS:  142.3 us  │  92.4 TFLOPS  │  612.0 GB/s
    cuBLAS:   105.2 us  │ 125.0 TFLOPS  │  828.4 GB/s
    ─────────────────────────────────────────
    Speedup:  1.35x
    NCU:      compute_bound  SM=62.3%  Occ=45.2%
```

### 输出路径

```
test/benchmark/ops/results/
├── latest.json              ← 软链接到最新
├── YYYY-MM-DD/hhmmss.json   ← 归档
└── ncu/*.ncu-rep            ← NCU 报告 (可 GUI 打开)
```

## 依赖与复用

- 导入 `test/test_utils.py` 的 `random_tensor`, `random_int_tensor`, `zero_tensor` 复用张量创建
- 正确性验证在 benchmark 中进行（跑完计时后用 `check_equal` 确认结果正确）
- 不复用现有 `benchmark()` 函数——重写为 CUDA Events 版本

## Verification

1. 每个 benchmark 脚本可独立运行并输出正确结果
2. `python run_all_benchmarks.py` 批量运行所有算子，生成 JSON
3. `python benchmark_linear.py --ncu quick` 成功生成 .ncu-rep 并解析出瓶颈分类
4. `bash ncu_profile.sh --op linear` 等价运行
5. 对比 LLAISYS vs cuBLAS 的 `linear` 结果在合理范围内
