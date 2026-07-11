# Operator Performance Benchmark

CUDA 算子性能测试 + 端到端推理性能对比（HF vs LLAISYS）。

## 快速开始

```bash
xmake f --nv-gpu=y -cv && xmake && xmake install
pip install -e ./python/

# 算子 benchmark
python test/benchmark/ops/benchmark_add.py --dtype f16

# 推理 benchmark
python test/benchmark/benchmark_infer.py --model <path> --warmup 1 --repeat 1
```

## 目录结构

```
test/benchmark/
├── README.md
├── infer_utils.py               ← 推理 benchmark 共享模块
├── benchmark_infer.py           ← 单请求串行推理 benchmark
├── benchmark_batch_infer.py     ← 并发请求推理 benchmark
├── ops/
│   ├── benchmark_harness.py     ← CUDA Event 计时 + 表格输出
│   ├── ncu_profiler.py          ← NCU 集成 + 动态指标发现 + 瓶颈分类
│   ├── benchmark_*.py           ← 10 个算子 benchmark
│   ├── run_all_benchmarks.py    ← 批量运行
│   └── results/                 ← JSON + .ncu-rep 输出
```

## 运算符子

| 脚本 | 算子 | 基准 |
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

## CLI 参数

### 通用参数

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--dtype` | `f16` `bf16` `f32` | `f16` | 数据类型 |
| `--warmup` | int | `10` | 计时前 warmup 次数 |
| `--repeat` | int | `10` (上限 20) | 计时迭代数 |
| `--use-ncu` | flag | — | NCU profiling（`--set full`），表格后自动 profile |
| `--example-index` | str | — | 指定 profile 的 shape 编号，逗号分隔如 `"0,3,7"` |
| `--output` | path | `results/` | JSON 输出目录 |

### benchmark_linear.py 独有

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--variant` | `all` `q_proj` `gate_proj` `down_proj` | `all` | 限制 Qwen2 投影层 |
| `--M` | int | `0` (全部) | 限制序列长度。`0` = M ∈ {1, 16, 64, 128, 512, 2048} |

### `--warmup` 与 `--repeat`

```
python test/benchmark/ops/benchmark_linear.py --warmup 10 --repeat 10
```

- warmup：跑 N 次完整算子调用 + sync。不计时。目的：稳定 GPU 频率，预热 cache
- repeat：跑 N 次（上限 20）。每次：分配新 tensor → CUDA Event start → 算子 → CUDA Event end → sync
- 报告值：**min** 为主值（CUTLASS 规范——噪声只增不减）

### `--use-ncu`

```bash
python test/benchmark/ops/benchmark_linear.py --use-ncu                        # 自动选最差 shape
python test/benchmark/ops/benchmark_linear.py --use-ncu --example-index "3"    # 只 profile #3
python test/benchmark/ops/benchmark_linear.py --use-ncu --example-index "0,7"  # profile #0 #7
```

- **流程**：先正常跑 benchmark → 输出表格 → 自动选 shape（speedup < 1.0 选最小，全 >= 1.0 选最接近 1.0）→ NCU `--set full` profile 子进程 → 输出瓶颈
- **结果**：`results/ncu_results.json` + `results/ncu/<op>_idx<N>.ncu-rep`
- **注意**：benchmark 跑两次（第一次表格，第二次 NCU 采集），NCU 固有限制

## 输出格式

### 终端（正常 benchmark）

```
  Correctness: PASS

  GPU: NVIDIA GeForce RTX 4060 Laptop GPU  |  CUDA: 12.8  |  Saved: results/run_.../results.json

  Operator: add
  ════════════════════════════════════════════════════════════════════════════
     shape        dtype  latency(us)  TFLOPS  baseline(us)  baseline(TF)  speedup
  ────────────────────────────────────────────────────────────────────────────
  0   1 x 3584     f16         16.3     0.0          28.5           0.0    1.75x
  1   16 x 3584    f16         43.0     0.0          12.3           0.0    0.29x
  ...
  ────────────────────────────────────────────────────────────────────────────
    f16  │ range: 9.4 — 136.0 us  │ mean: 44.6 us  │ speedup: 0.21x — 1.75x
```

- 左起第一列为 shape 编号，可直接用于 `--example-index`
- speedup = baseline / LLAISYS。> 1.0 = LLAISYS 更快

### 终端（`--use-ncu`）

```
  [表格输出同上]
  ────────────────────────────────────────────────────────────────────────────
    f16  │ range: 154.6 — 154.6 us  │ mean: 154.6 us  │ speedup: 1.01x — 1.01x

  NCU #3: latency_bound  SM=43.8%  DRAM=0.0%  Occ=23.7%
  NCU results saved: /home/.../results/ncu_results.json
```

### JSON

**基准结果**：`results/run_YYYYmmdd_HHMMSS/results.json`（软链接 `results/latest.json`）

**NCU 结果**：`results/ncu_results.json`

```json
[{
  "operator": "linear",
  "shape_index": 3,
  "report_file": "results/ncu/linear_idx3.ncu-rep",
  "kernel_count": 2,
  "total_time_us": 154.6,
  "bottleneck": "latency_bound",
  "sm_throughput_pct": 43.8,
  "dram_throughput_pct": 0.0,
  "occupancy_pct": 23.7,
  "timestamp": "2026-07-11T23:35:14"
}]
```

## NCU Profiling

指标通过 `ncu --list-metrics` 动态发现（GPU 型号缓存，24h TTL），使用前缀匹配自动适配不同 GPU 架构。

瓶颈分类规则：

| 条件 | 分类 |
|---|---|
| DRAM > 60% 且 SM < 20% | `memory_bound` |
| SM > 60% 且 DRAM < 20% | `compute_bound` |
| 两者 < 30% | `launch_occupancy_bound` |
| 其他 | `latency_bound` |

**注意**：部分 GPU（如 RTX 4060 Ada Lovelace）不提供 `dram__throughput` 百分比指标，此时 DRAM 为 0.0% 且瓶颈偏向 `latency_bound`。这是该 GPU 的硬件计数器限制，不影响 SM 和 Occupancy 数据的准确性。

## 推理 Benchmark

端到端推理性能对比（HuggingFace vs LLAISYS），支持单请求串行和并发多请求两种模式。

### CLI

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--device` | `nvidia` | `nvidia` | CUDA 仅用 |
| `--model` | path | — | 模型路径（本地或自动下载 HF） |
| `--seed` | int | `42` | 随机种子 |
| `--max_steps` | int | `10` | 每请求最大生成 token 数 |
| `--top_p` | float | `1.0` | 采样参数 |
| `--top_k` | int | `1` | 采样参数 |
| `--temperature` | float | `1.0` | 采样参数 |
| `--num_prompts` | int | `12` | 测试 prompt 数量（从 short/medium/long 三层均匀采样） |
| `--warmup` | int | `3` | warmup 迭代数（LLAISYS 用 disposable prompt 避免 KV cache 污染） |
| `--repeat` | int | `5` | 计时迭代数（对 LLAISYS 无意义：同 prompt 二次调用命中 KV cache） |
| `--test` | flag | — | greedy 解码 + token 级正确性验证 |
| `--output` | path | `test/benchmark/results` | JSON 输出目录 |

### 单请求串行

```bash
python test/benchmark/benchmark_infer.py --model <path>
python test/benchmark/benchmark_infer.py --model <path> --test   # 正确性验证
```

逐 prompt 串行调用 `model.generate()`，测量每个请求的延迟。输出：

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Prompts: 6  |  Mode: single

  ═══════════════════════════════════════════════════════════════════════════
  prompt    input(token)  LLAISYS_output(token)  HF_output(token)  LLAISYS_latency(ms)  HF_latency(ms)  speedup
  ───────────────────────────────────────────────────────────────────────────
  short_0            13                    10                10               205.4            221.9    1.08x
  medium_0           35                    10                10               202.6            224.7    1.11x
  long_0            186                    10                10               249.3            236.3    0.95x
  ═══════════════════════════════════════════════════════════════════════════

  Metric          LLAISYS           HF
  ─────────────────────────────────────
  avg latency      235.5 ms     224.7 ms
  p50 latency      239.3 ms     224.7 ms
  p90 latency      249.3 ms     235.9 ms
  TTFT(avg)              —       19.9 ms
  TPOT            23.61 ms     22.53 ms
  Throughput       42.4 t/s     44.4 t/s
```

- speedup 基于 **throughput**（tokens/s），考虑输出长度差异
- prompt 从 short/medium/long 三层采样，`{tier}_{index}` 标签

### 并发多请求

```bash
python test/benchmark/benchmark_batch_infer.py --model <path>
```

N 个线程同时调用 `model.generate()`，LLAISYS scheduler 内部自动组 batch。HF 用 padded batch 对比。

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Prompts: 6  |  Mode: concurrent

  ═══════════════════════════════════════════════════════════════════════════
  prompt    input(token)  LLAISYS_output(token)  HF_output(token)  LLAISYS_latency(ms)  HF_latency(ms)  speedup
  ───────────────────────────────────────────────────────────────────────────
  (6 prompts)    532              60                  60               450.2            580.3    1.29x
  ═══════════════════════════════════════════════════════════════════════════
  Throughput:  LLAISYS 133.2 t/s  HF 103.4 t/s
```

并发表不显示 per-request 延迟分位数（`time.perf_counter()` 无法拆分到单请求）。

### JSON 输出

`results/run_YYYYmmdd_HHMMSS/results.json`：

```json
{
  "environment": {"gpu": "RTX 4060", "cuda": "12.8", "timestamp": "..."},
  "config": {"mode": "single", "num_prompts": 12, "seed": 42, ...},
  "results": {
    "hf": {"elapsed_s": ..., "tpot_ms": ..., "throughput": ..., "lat_avg": ..., "lat_p50": ..., "lat_p90": ...},
    "llaisys": {...},
    "speedup": {"wall_clock": 1.23, "tpot": 1.24}
  }
}
```

## 批量运行

```bash
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op all
python test/benchmark/ops/run_all_benchmarks.py --dtype f16 --op linear,rope,rms_norm
```

## Shape 覆盖

基于 Qwen2-7B 模型维度（hs=3584, di=18944, nh=28, nkvh=4, dh=128, vocab=151936）：

| 算子 | M | N/K |
|---|---|---|
| add / argmax / rearrange | 1..2048 | ×3584 |
| embedding | 1..2048 | vocab=151936 |
| linear q_proj | 1..2048 | N=3584, K=3584 |
| linear gate/up | 1..2048 | N=18944, K=3584 |
| linear down | 1..2048 | N=3584, K=18944 |
| rms_norm | 1..2048 | ×3584 |
| rope | 1..2048 | nh=28/4, hd=128 |
| swiglu | 1..2048 | ×18944 |
| self_attention | 1..512 | total_len: 64..1024 |
| topk | (3584,)..(128,32000) | k=10 |
