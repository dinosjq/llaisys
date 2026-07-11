# summarize_run API Fix — Design Spec

Date: 2026-07-12

## Context

`summarize_run` 试图从 `results: list[tuple]` 反推 input/output tokens 和 per-request latency，但各调用方传入的 tuple 格式不一致：
- benchmark_infer.py: `(tokens, tokens, e_us)` → output_tokens=0
- benchmark_batch_infer.py: `(r[0], r[1], 0)` for HF, `(r[0], r[0], 0)` for LLAISYS → latencies=0

## Fix

将 `summarize_run` 改为接受调用方预计算的聚合值：

```python
def summarize_run(name, elapsed_s, num_prompts,
                  total_input_tokens, total_output_tokens,
                  latencies_us=None) -> dict
```

- `num_prompts`：prompt 数量
- `total_input_tokens` / `total_output_tokens`：由调用方精确计算
- `latencies_us`：可选。传入 per-request CUDA Event 延迟列表则计算 P50/P90；并发模式传 None 则跳过

## Caller changes

**benchmark_infer.py**：有 CUDA Event 延迟

```python
input_tokens = sum(len(_apply_chat(tokenizer, p)) for p in prompts)
output_tokens = sum(len(r[0]) - input_lengths[i] for i, r in enumerate(results))
latencies = [r[2] for r in results]  # CUDA Event elapsed_us
hf_stats = summarize_run("HF", elapsed_s, len(prompts), input_tokens, output_tokens, latencies)
```

**benchmark_batch_infer.py**：无 per-request 延迟

```python
input_tokens = sum(len(_apply_chat(tokenizer, p)) for p in prompts)
# HF: r is (input_ids, full_ids)
output_tokens = sum(len(r[1]) - len(r[0]) for r in results)
hf_stats = summarize_run("HF", elapsed_s, len(prompts), input_tokens, output_tokens, latencies_us=None)
# LLAISYS: 需要从 llaisys_infer 结果计算 output_tokens
```

## Files

- `test/benchmark/infer_utils.py`: `summarize_run` 签名 + 实现
- `test/benchmark/benchmark_infer.py`: 调用方更新
- `test/benchmark/benchmark_batch_infer.py`: 调用方更新
