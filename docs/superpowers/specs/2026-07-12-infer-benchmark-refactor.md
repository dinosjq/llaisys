# Inference Benchmark Refactor — Design Spec (v2)

Date: 2026-07-12

## Context

`test/benchmark/benchmark_infer.py` 和 `benchmark_batch_infer.py` 由原 `test/test_infer.py`、`test/test_infer_batch.py` 移入。两份代码互相重复，helper 函数与 `test/test_utils.py` 重复，无 warmup/repeat，计时用 `time.perf_counter()`，`load_hf_model` 行为不一致。

## Goals

- 消除代码重复：提取 `test/benchmark/infer_utils.py`
- 性能测试 + `--test` 模式做 greedy 正确性验证
- CUDA 仅用 (CPU 不支持)
- warmup / repeat 标准化流程
- 保留 `--seed` 可重现性
- JSON 输出 + 终端表格

## Files

```
test/benchmark/
├── infer_utils.py
│   ├── torch_device / llaisys_device     ← from test_utils
│   ├── load_hf_model                     ← 统一实现：device_map + pad_token
│   ├── load_llaisys_model
│   ├── build_prompt_bank(num, context_repeat) → list[str]
│   ├── hf_infer(prompt, ...)             ← 单 prompt, CUDA Event
│   ├── llaisys_infer(prompt, ...)        ← 单 prompt, CUDA Event
│   ├── hf_batch_infer(prompts, ...)      ← padded batch
│   ├── llaisys_concurrent_infer(...)     ← 线程并发, time.perf_counter
│   ├── summarize_run / print_speedup
│   └── save_json(results, path)
│
├── benchmark_infer.py                    ← 单请求串行基准
│   └── 串行遍历 prompts，逐个 hf_infer / llaisys_infer
│
└── benchmark_batch_infer.py              ← 并发请求基准
    └── N 线程同时调 model.generate()，scheduler 自动组 batch
```

## load_hf_model 统一实现

```python
def load_hf_model(model_path=None, device_name="nvidia"):
    # 1. 下载/加载模型 → to GPU via device_map
    # 2. 添加 <|pad|> token（HF batch 需要）
    # 3. resize_token_embeddings
    # 返回 (tokenizer, model, model_path)
```

## 两种计时策略

| 场景 | 方法 | 原因 |
|---|---|---|
| 单 prompt（hf_infer / llaisys_infer） | `torch.cuda.Event` | 单 stream，GPU 时间精确 |
| 并发（llaisys_concurrent_infer） | `time.perf_counter()` | 多线程 + C++ 内部 stream，CUDA Event 不适用 |
| 批处理（hf_batch_infer） | `torch.cuda.Event` | 单 forward pass |

## CLI

```
--device nvidia                          # 仅 CUDA
--model <path>
--seed 42
--max_steps 32
--top_p 1.0 --top_k 1 --temperature 1.0
--num_prompts 12 --context_repeat 2
--warmup 3 --repeat 5
--test                                   # greedy 解码 + token 级正确性验证
--output results/                        # JSON 输出目录
```

`--test` 模式：top_p=1.0, top_k=1, temperature=1.0，逐 token 对比 HF vs LLAISYS。

## 输出格式

### 终端（两个脚本统一风格）

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Prompts: 12  |  Mode: single

  Metric            HF          LLAISYS     Speedup
  ───────────────────────────────────────────────────
  Latency(avg)      245.3 ms    198.7 ms    1.23x
  Latency(p50)      230.1 ms    185.2 ms    1.24x
  Latency(p90)      312.4 ms    245.1 ms    1.27x
  TPOT              12.5 ms     10.1 ms     1.24x
  Throughput        80.0 tok/s  99.0 tok/s  1.24x
```

并发模式额外显示 `Mode: concurrent` 和 `Total elapsed`：

```
  Model: Qwen2-1.5B  |  GPU: RTX 4060  |  Prompts: 12  |  Mode: concurrent

  Metric                HF(batch)   LLAISYS(concurrent)  Speedup
  ─────────────────────────────────────────────────────────────────
  Total elapsed         1.23 s       0.98 s                1.26x
  Latency(p50)          1.18 s       0.92 s                1.28x
  Latency(p90)          1.22 s       0.95 s                1.28x
  TPOT                  8.2 ms       6.5 ms                1.26x
  Throughput            450 tok/s    565 tok/s             1.26x
```

### JSON

```
results/run_YYYYmmdd_HHMMSS/results.json
```

```json
{
  "environment": {"gpu": "...", "model": "Qwen2-1.5B", "timestamp": "..."},
  "config": {"mode": "single", "num_prompts": 12, "max_steps": 32, ...},
  "results": {
    "hf": {"elapsed_s": ..., "metrics": {...}},
    "llaisys": {"elapsed_s": ..., "metrics": {...}},
    "speedup": {"wall_clock": 1.23, "tpot": 1.24, ...}
  }
}
```

## Verification

```bash
python test/benchmark/benchmark_infer.py --model <path> --test --warmup 1 --repeat 2
python test/benchmark/benchmark_batch_infer.py --model <path> --test --warmup 1 --repeat 2
```

预期：`--test` 模式下 token 序列完全匹配，性能数据非零。
