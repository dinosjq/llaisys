# Inference Operator Profiling

端到端推理 pipeline 的逐算子耗时 profiling 系统。每次运行 profile 一个 prompt。

## 快速开始

```bash
xmake f --nv-gpu=y --profiling=y -c && xmake && xmake install

# 默认中等长度 prompt
python test/profile/profile_model.py --model <path>

# 指定输入长度档位
python test/profile/profile_model.py --model <path> --prompt-len 500

# 自定义 prompt
python test/profile/profile_model.py --model <path> --prompt "Explain quantum computing."
```

## CLI

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | (required) | 模型路径 |
| `--max_steps` | 1025 | 最大生成 token 数 |
| `--prompt-len` | 50 | 预设 prompt 长度档位 (10/50/100/500) |
| `--prompt` | — | 自定义 prompt（覆盖 --prompt-len） |

## 采样点

decode step: {0(prefill), 1, 32, 64, 128, 256, 512, 1024}

## 输出格式

```
  input_token: 23   output_token: 32   elapsed: 1.84s
  throughput: 17.4 tok/s   TTFT: 258.50 ms   TPOT: 57.46 ms

  period: prefill   step: 0   total_ms: 253.81
    operator          avg_ms    count    total_ms     %
    ───────────────────────────────────────────────────
    linear:q_proj       4.17       28      116.75   46.0
    embed              82.44        1       82.44   32.5
    linear:k_proj       0.32       28        8.89    3.5
    ...

  period: decode   step: 128   total_ms: 51.82
    operator          avg_ms    count    total_ms     %
    ───────────────────────────────────────────────────
    paged_attn          0.15       28        4.17    8.1
    ...
```

- `throughput` / `TPOT` 基于 wall-clock `time.perf_counter()`（非采样 step 零开销）
- per-step `total_ms` 含 CUDA event 测量开销，用于**相对瓶颈分析**
- 绝对值以 `benchmark_infer.py` 的 profiling=n 结果为准
- `TTFT` = prefill step total_ms
- `%` = operator total_ms / step total_ms × 100
- JSON 输出到 `test/profile/results/profile_YYYYmmdd_HHMMSS.json`
