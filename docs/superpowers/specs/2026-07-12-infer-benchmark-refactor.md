# Inference Benchmark Refactor — Design Spec

Date: 2026-07-12

## Context

`test/benchmark/benchmark_infer.py` 和 `benchmark_batch_infer.py` 由原 `test/test_infer.py`、`test/test_infer_batch.py` 移入。两份代码互相重复，helper 函数与 `test/test_utils.py` 重复，没有 warmup/repeat，计时用 `time.perf_counter()` 而非 CUDA Event。

## Goals

- 消除代码重复：提取 `test/benchmark/infer_utils.py`
- 纯性能测试（不做 token 级正确性对比）
- CUDA Event 计时
- 标准化 warmup/repeat 流程
- 表格风格输出（与 operator benchmark 一致）
- LLAISYS 并发模型正确：线程同时调 `model.generate()`，scheduler 内部组 batch

## Files

```
test/benchmark/
├── infer_utils.py              ← 新建
│   ├── torch_device / llaisys_device   ← from test_utils
│   ├── load_hf_model / load_llaisys_model
│   ├── build_prompt_bank
│   ├── hf_infer / llaisys_infer         ← 单次调用 + CUDA Event
│   ├── hf_batch_infer                   ← padded batch
│   ├── llaisys_concurrent_infer         ← 线程并发
│   ├── summarize_run / print_speedup
│
├── benchmark_infer.py           ← 单请求串行
│   └── 串行跑 N 个 prompt → 延迟分布
│
└── benchmark_batch_infer.py     ← 并发请求
    └── N 线程同时调用 → 吞吐 + batch size
```

## CLI

```
--device nvidia
--model <path>
--max_steps 32
--top_p 1.0 --top_k 1 --temperature 1.0
--num_prompts 12 --context_repeat 2
--warmup 3 --repeat 5
```

## Output Format

### benchmark_infer.py

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Prompts: 12

  Metric          HF          LLAISYS     Speedup
  Latency(avg)    245.3 ms    198.7 ms    1.23x
  Latency(p50)    230.1 ms    185.2 ms    1.24x
  Latency(p90)    312.4 ms    245.1 ms    1.27x
  TPOT            12.5 ms     10.1 ms     1.24x
  Throughput      80.0 tok/s  99.0 tok/s  1.24x
```

### benchmark_batch_infer.py

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Concurrent: 12

  Metric          HF(batch)    LLAISYS(concurrent)  Speedup
  Total elapsed   1.23 s       0.98 s                1.26x
  Throughput      450 tok/s    565 tok/s             1.26x
  TPOT            8.2 ms       6.5 ms                1.26x
  Latency(p50)    1.18 s       0.92 s                1.28x
  Latency(p90)    1.22 s       0.95 s                1.28x
```

## Verification

```bash
python test/benchmark/benchmark_infer.py --model <path> --warmup 1 --repeat 2
python test/benchmark/benchmark_batch_infer.py --model <path> --warmup 1 --repeat 2
```
