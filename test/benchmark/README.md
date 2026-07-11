# Operator Performance Benchmark

基于 CUDA Event 的算子性能测试系统，集成 NCU (NVIDIA Nsight Compute) profiling。

## 快速开始

```bash
# 编译 C++ 后端（如未编译）
xmake f --nv-gpu=y -cv && xmake && xmake install

# 安装 Python 包
pip install -e ./python/

# 运行单个算子 benchmark
python test/benchmark/ops/benchmark_linear.py --dtype f16

# 批量运行全部算子
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op all
```

## 目录结构

```
test/benchmark/
├── README.md                    ← 本文档
├── IMPLEMENTATION_REPORT.md     ← 实现报告
└── ops/
    ├── benchmark_harness.py     ← 共享计时框架
    ├── ncu_profiler.py          ← NCU 集成模块
    ├── benchmark_linear.py      ← 各算子 benchmark（12 个）
    ├── run_all_benchmarks.py    ← 批量运行入口
    ├── ncu_profile.sh           ← NCU Shell 包装
    └── results/                 ← 输出目录
```

## 运算符子

| 脚本 | 算子 | 基准对比 |
|---|---|---|
| `benchmark_add.py` | 逐元素加法 | PyTorch CUDA |
| `benchmark_argmax.py` | 最大值归约 | PyTorch CUDA |
| `benchmark_embedding.py` | Token 嵌入 | PyTorch CUDA |
| `benchmark_rearrange.py` | 张量重排 | PyTorch CUDA |
| `benchmark_linear.py` | 矩阵乘法 | cuBLAS |
| `benchmark_rms_norm.py` | RMS 归一化 | PyTorch |
| `benchmark_rope.py` | 旋转位置编码 | PyTorch |
| `benchmark_swiglu.py` | SwiGLU 激活 | PyTorch |
| `benchmark_self_attention.py` | 自注意力 | Flash-Attention |
| `benchmark_topk.py` | Top-K 选择 | PyTorch CUDA |
| `benchmark_paged_attention.py` | 分页注意力 | 无（自定义算子） |
| `benchmark_kv_cache_move.py` | KV Cache 搬运 | 无（自定义算子） |

## CLI 参数

### 所有 benchmark 脚本通用参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dtype` | `f16` `bf16` `f32` | `f16` | 张量数据类型 |
| `--warmup` | int | `10` | 计时前的 warmup 迭代数。不计时，用于稳定 GPU 频率和预热缓存 |
| `--repeat` | int | `100` | 正式计时迭代数。每次迭代重新分配 tensor 以击败 L2 cache，用 CUDA Event 记录 GPU 执行时间 |
| `--ncu` | `quick` `detailed` `full` | — | 启用 NCU profiling 模式。不传值默认 `detailed` |
| `--output` | path | `test/benchmark/ops/results` | JSON 结果输出目录 |

### 指定 shape 范围（仅 benchmark_linear.py）

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--variant` | `all` `q_proj` `gate_proj` `down_proj` | `all` | 限制测试的 Qwen2 投影层类型 |
| `--M` | int | `0`（全部 6 个 M） | 限制序列长度。`0` = 扫全部 M ∈ {1, 16, 64, 128, 512, 2048} |

### 参数说明详解

**`--warmup` 与 `--repeat`**

```
python benchmark_linear.py --warmup 10 --repeat 100
```

- warmup：跑 10 次完整的算子调用 + CUDA 同步。不计时，不收集数据。目的：稳定 GPU 时钟频率，预热 L1/L2 cache
- repeat：跑 100 次。每次：分配新 tensor → record CUDA Event start → 执行算子 → record CUDA Event end → sync。收集 100 个 GPU 时间样本
- 报告值：**min** 作为主值（CUTLASS 规范——噪声只增不减，min 最接近硬件极限）

**`--ncu`**

```
python benchmark_linear.py --ncu           # 等价于 --ncu detailed
python benchmark_linear.py --ncu quick     # SpeedOfLight 快速诊断
python benchmark_linear.py --ncu full      # 全量指标
```

- 启用后：benchmark 先跑完整循环 → 输出表格 → 然后用 subprocess 启动 NCU → NCU 再次启动同一个脚本作为子进程 → 采集 GPU 硬件计数器 → 解析结果 → 在表格中加入瓶颈分类
- NCU 的 `--launch-skip` 和 `--launch-count` 由 `ncu_profiler.py` 的 `NCUConfig` 管理，用户不可见

## 输出格式

### 终端输出

```
  Operator: linear
  ══════════════════════════════════════════════════════════════════════
    shape                       dtype  latency(us)  TFLOPS  baseline(us)  baseline(TF)  speedup
  ────────────────────────────────────────────────────────────────────────
    1 x 3584 x 3584 x q_proj    f16         234.5     0.1         91.1           0.3    2.57x
    16 x 3584 x 3584 x q_proj   f16          65.5     6.3        106.5           3.9    1.62x
    ...
  ────────────────────────────────────────────────────────────────────────
    f16  │ range: 65.5 — 2124.8 us  │ mean: 765.2 us  │ speedup: 0.59x — 2.57x
```

- 表头：shape / dtype / latency(us) / TFLOPS / baseline(us) / baseline(TF) / speedup
- speedup = baseline / LLAISYS。> 1.0 = LLAISYS 更快，< 1.0 = baseline 更快
- footer：汇总所有 shape 的 range / mean / speedup range

### JSON 输出

每次运行在 `results/run_YYYYmmdd_HHMMSS/results.json`：

```json
{
  "environment": {
    "cuda_version": "12.8",
    "gpu_model": "NVIDIA GeForce RTX 4060 Laptop GPU",
    "timestamp": "20260711_200627"
  },
  "results": [{
    "operator": "linear",
    "dtype": "f16",
    "shape": {"M": 128, "N": 3584, "K": 3584, "variant": "q_proj"},
    "latency_us": 151.6,
    "latency_stats": {"min": 151.6, "median": 152.1, "mean": 153.2, "stddev": 1.8, "p99": 158.4},
    "bandwidth_GBs": 89.3,
    "TFLOPS": 21.7,
    "baseline_name": "cuBLAS",
    "baseline_latency_us": 143.4,
    "baseline_TFLOPS": 22.9,
    "speedup": 0.95,
    "ncu_set": "detailed",
    "ncu_report_file": "results/ncu/linear_detailed.ncu-rep",
    "ncu_bottleneck": "compute_bound",
    "ncu_sm_throughput_pct": 62.3,
    "ncu_dram_throughput_pct": 38.1,
    "ncu_occupancy_pct": 45.2
  }]
}
```

最新的运行通过 `results/latest.json` 软链接访问。

## NCU Profiling

### Shell 包装（推荐）

```bash
bash test/benchmark/ops/ncu_profile.sh --op linear --set default
bash test/benchmark/ops/ncu_profile.sh --op rope --set detailed
```

参数：`--op` 算子名、`--set` NCU 章节集（`default`/`detailed`/`full`）、`--dtype` 数据类型。

### Python CLI

```bash
python test/benchmark/ops/benchmark_linear.py --ncu quick --M 128 --variant q_proj
```

NCU 采集的硬件计数器：

| 指标 | 含义 |
|---|---|
| `gpu__time_duration.sum` | GPU 执行时间 (ns) |
| `dram__bytes_read.sum` | HBM 读取字节 |
| `dram__bytes_write.sum` | HBM 写入字节 |
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM 吞吐率（%峰值） |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM 吞吐率（%峰值） |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | Warp 占用率（%峰值） |

瓶颈分类规则：

| 条件 | 分类 |
|---|---|
| HBM > 60% 且 SM < 20% | `memory_bound` |
| SM > 60% 且 HBM < 20% | `compute_bound` |
| 两者 < 30% | `launch_occupancy_bound` |
| 其他 | `latency_bound` |

## 批量运行

```bash
# 全部算子
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op all

# 指定算子
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op linear,rope,rms_norm

# 带 NCU
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op all --ncu quick
```

## Shape 覆盖

基于 Qwen2-7B 模型实际维度（`hidden_size=3584`, `intermediate_size=18944`, `num_heads=28`, `num_kv_heads=4`, `head_dim=128`, `vocab_size=151936`）：

| 算子 | M 范围 | N/K 范围 |
|---|---|---|
| add / argmax / rearrange | 1, 16, 64, 128, 512, 2048 | ×3584 |
| argmax 额外 | — | ×18944 |
| embedding | seqlen: 1..2048 | vocab=151936, dim=3584 |
| linear q_proj | 1..2048 | N=3584, K=3584 |
| linear gate_proj | 1..2048 | N=18944, K=3584 |
| linear down_proj | 1..2048 | N=3584, K=18944 |
| rms_norm | 1..2048 | ×3584 |
| rope | seqlen: 1..2048 | nh=28/4, hd=128 |
| swiglu | 1..2048 | ×18944 |
| self_attention | seqlen: 1..512 | total_len: 64..1024 |
| topk | (3584,)..(128,32000) | k=10 |
| paged_attention | seqlen: 16..512 | token_num=64 |
| kv_cache_move | batch: 1..8 | max_seq_len: 64..512 |

## 常见用例

```bash
# 查看 linear 算子在 q_proj 上 M=128 的性能
python test/benchmark/ops/benchmark_linear.py --dtype f16 --M 128 --variant q_proj

# 看全部 18 个 shape 组合（默认行为）
python test/benchmark/ops/benchmark_linear.py --dtype f16

# 快速 NCU 诊断（仅一个 shape）
python test/benchmark/ops/benchmark_linear.py --ncu quick --M 128 --variant q_proj --warmup 5 --repeat 5

# 生成 .ncu-rep 用于 GUI 深度分析
bash test/benchmark/ops/ncu_profile.sh --op linear --set detailed

# 只看结果不跑 NCU
python test/benchmark/ops/benchmark_linear.py --dtype f16 --M 512
```
