# Inference Benchmark Refactor — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace duplicated, inconsistent inference benchmark code with a shared `infer_utils.py` module and two clean entry points.

**Architecture:** `infer_utils.py` provides model loading, inference helpers, metrics aggregation, and JSON output. `benchmark_infer.py` runs serial single-prompt benchmarks. `benchmark_batch_infer.py` runs concurrent multi-request benchmarks. Both share identical CLI and output format.

**Tech Stack:** Python 3.9+, PyTorch, HuggingFace Transformers, LLAISYS ctypes

## Global Constraints

- CUDA only (no CPU support)
- Dual timing: `torch.cuda.Event` for single-prompt, `time.perf_counter()` for concurrent
- `load_hf_model` must use `device_map` + add `<|pad|>` token + `resize_token_embeddings`
- `--test` mode: greedy decoding (top_p=1, top_k=1, temp=1.0), token-level comparison
- `--seed 42` for reproducibility
- `--output` JSON directory, default `results/`
- Import `torch_device`, `llaisys_device` from `test.test_utils`
- Warmup/repeat: `--warmup 3 --repeat 5`

---

### Task 1: Create infer_utils.py — shared module

**Files:**
- Create: `test/benchmark/infer_utils.py`
- Read for reference: `test/test_infer.py`, `test/test_utils.py`

**Interfaces:**
- Produces:
  - `load_hf_model(model_path, device_name) -> (tokenizer, model, model_path)`
  - `load_llaisys_model(model_path, device_name) -> Qwen2`
  - `build_prompt_bank(num_prompts, context_repeat) -> list[str]`
  - `hf_infer(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature) -> (tokens, elapsed_us)`
  - `llaisys_infer(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature) -> (tokens, elapsed_us)`
  - `hf_batch_infer(prompts, tokenizer, model, ...) -> (results: list, elapsed_us: float)`
  - `llaisys_concurrent_infer(prompts, tokenizer, model, ...) -> (results: list, elapsed_s: float)`
  - `summarize_run(name, elapsed_s, results) -> dict`
  - `print_speedup(hf_stats, ll_stats) -> None`
  - `save_json(results, output_dir) -> str`
  - `format_table(hf_stats, ll_stats, mode) -> str`

- [ ] **Step 1: Write infer_utils.py**

```python
"""Shared utilities for inference benchmarks."""

import gc
import json
import os
import threading
import time
import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

import llaisys
from test_utils import torch_device, llaisys_device

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated!")


# ── Model Loading ─────────────────────────────────────────────────

def load_hf_model(model_path=None, device_name="nvidia"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path and os.path.isdir(model_path):
        print(f"Loading HF model from local: {model_path}")
    else:
        print(f"Downloading HF model: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name), trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model, model_path


def load_llaisys_model(model_path, device_name):
    return llaisys.models.Qwen2(model_path, llaisys_device(device_name))


# ── Prompt Construction ──────────────────────────────────────────

def build_prompt_bank(num_prompts, context_repeat):
    base_prompts = [
        "Explain what batch inference is and why it matters for serving large language models.",
        "Write a concise comparison between prefill and decode phases in autoregressive generation.",
        "Summarize the tradeoffs between latency, throughput, and batch size in inference systems.",
        "Give a step-by-step explanation of how KV cache helps speed up transformer decoding.",
        "Describe three common bottlenecks in GPU inference pipelines and how to measure them.",
        "Draft a short checklist for validating a new model backend before releasing it to users.",
        "What are the risks of measuring performance with too few prompts or too few generated tokens?",
        "Produce a short technical note about concurrency, scheduling, and tail latency in model serving.",
    ]
    extra_contexts = [
        "Use concrete terms and keep the answer focused on engineering tradeoffs.",
        "Include at least one practical example or metric in the response.",
        "Mention how the answer would change under CPU and GPU execution.",
        "Keep the output dense enough to exercise generation cost, not just a one-line response.",
    ]
    prompts = []
    for idx in range(num_prompts):
        topic = base_prompts[idx % len(base_prompts)]
        extra = extra_contexts[idx % len(extra_contexts)]
        context_lines = [
            f"Context line {ri + 1}: this benchmark should stress input handling, decoding, and batching behavior."
            for ri in range(context_repeat)
        ]
        prompt = f"{topic}\n\n{extra}"
        if context_lines:
            prompt += "\n\n" + "\n".join(context_lines)
        prompts.append(prompt)
    return prompts


# ── Single-Prompt Inference (CUDA Event timing) ──────────────────

def _apply_chat(tokenizer, prompt):
    content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return tokenizer.encode(content)


def hf_infer(prompt, tokenizer, model, max_new_tokens=128,
             top_p=0.8, top_k=50, temperature=0.8):
    input_ids = _apply_chat(tokenizer, prompt)
    inputs = torch.tensor([input_ids], device=model.device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_new_tokens, top_k=top_k,
            top_p=top_p, temperature=temperature, use_cache=True,
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0
    tokens = outputs[0].tolist()
    return tokens, elapsed_us


def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128,
                  top_p=0.8, top_k=50, temperature=0.8):
    input_ids = _apply_chat(tokenizer, prompt)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        top_k=top_k, top_p=top_p, temperature=temperature,
    )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0
    return result, elapsed_us


# ── Batch / Concurrent Inference ─────────────────────────────────

def hf_batch_infer(prompts, tokenizer, model, max_new_tokens=128,
                   top_p=0.8, top_k=50, temperature=0.8):
    """Padded batch forward pass.  Returns (results, elapsed_us)."""
    contents = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False,
        ) for p in prompts
    ]
    enc = tokenizer(contents, return_tensors="pt", padding=True,
                    truncation=True).to(model.device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(
            **enc, max_new_tokens=max_new_tokens, top_k=top_k,
            top_p=top_p, temperature=temperature, use_cache=True,
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0

    input_lens = enc.attention_mask.sum(dim=1).tolist()
    max_input_len = enc.input_ids.shape[1]
    results = []
    for idx in range(len(prompts)):
        input_ids = outputs[idx, :input_lens[idx]].tolist()
        full = outputs[idx].tolist()
        gen_part = full[max_input_len:]
        while gen_part and gen_part[-1] == tokenizer.pad_token_id:
            gen_part.pop()
        results.append((input_ids, input_ids + gen_part))
    return results, elapsed_us


def llaisys_concurrent_infer(prompts, tokenizer, model, max_new_tokens=128,
                             top_p=0.8, top_k=50, temperature=0.8):
    """Threaded concurrent requests. Returns (results, elapsed_s)."""
    results = [None] * len(prompts)

    def _worker(idx, prompt):
        results[idx] = llaisys_infer(
            prompt, tokenizer, model, max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k, temperature=temperature,
        )

    threads = []
    start = time.perf_counter()
    for i, p in enumerate(prompts):
        t = threading.Thread(target=_worker, args=(i, p))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed_s = time.perf_counter() - start
    torch.cuda.synchronize()
    return results, elapsed_s


# ── Metrics ──────────────────────────────────────────────────────

def summarize_run(name, elapsed_s, results):
    """Compute aggregate metrics from a list of (tokens, elapsed) tuples."""
    input_tokens  = sum(len(r[0]) for r in results)
    output_tokens = sum(len(r[1]) - len(r[0]) if len(r) > 1 and isinstance(r[1], list)
                        else 0 for r in results)
    latencies = [r[2] if len(r) > 2 else elapsed_s for r in results]  # us or s
    if output_tokens == 0:
        output_tokens = 1
    tpot_ms = (elapsed_s * 1000.0 / output_tokens) if isinstance(elapsed_s, float) else 0
    throughput = output_tokens / elapsed_s if elapsed_s > 0 else 0
    sorted_lat = sorted(latencies)
    n = len(sorted_lat)
    return {
        "name": name,
        "elapsed_s": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tpot_ms": tpot_ms,
        "throughput": throughput,
        "lat_avg": sum(latencies) / n if n else 0,
        "lat_p50": sorted_lat[n // 2] if n else 0,
        "lat_p90": sorted_lat[max(int(n * 0.9) - 1, 0)] if n else 0,
    }


def print_speedup(hf_stats, ll_stats):
    if ll_stats["elapsed_s"] <= 0:
        return
    s_wall = hf_stats["elapsed_s"] / ll_stats["elapsed_s"]
    s_tpot = hf_stats["tpot_ms"] / ll_stats["tpot_ms"] if ll_stats["tpot_ms"] > 0 else 0
    print(f"\n  Wall-clock speedup: {s_wall:.2f}x")
    print(f"  TPOT improvement:   {s_tpot:.2f}x")


def format_table(hf_stats, ll_stats, mode="single"):
    """Unified terminal table output."""
    def _ms(v):
        return v * 1000.0 if mode == "concurrent" else v
    def _fmt(v, unit="ms"):
        return f"{v:.1f} {unit}"

    lines = [
        f"\n  ┌{'─' * 66}┐",
    ]
    label = "HF(batch)" if mode == "concurrent" else "HF"
    ll_label = "LLAISYS(concurrent)" if mode == "concurrent" else "LLAISYS"

    rows = [
        ("Latency(avg)", "ms", hf_stats.get("lat_avg", 0),
         ll_stats.get("lat_avg", 0)),
        ("Latency(p50)", "ms",
         hf_stats.get("lat_p50", 0), ll_stats.get("lat_p50", 0)),
        ("Latency(p90)", "ms",
         hf_stats.get("lat_p90", 0), ll_stats.get("lat_p90", 0)),
        ("TPOT", "ms", hf_stats["tpot_ms"], ll_stats["tpot_ms"]),
        ("Throughput", "tok/s", hf_stats["throughput"], ll_stats["throughput"]),
    ]
    if mode == "concurrent":
        rows.insert(0, ("Total elapsed", "s",
                        hf_stats["elapsed_s"], ll_stats["elapsed_s"]))

    for metric, unit, hf_v, ll_v in rows:
        sp = hf_v / ll_v if ll_v > 0 else 0
        lines.append(
            f"  │ {metric:<18s} {_fmt(hf_v, unit):>12s}  {_fmt(ll_v, unit):>12s}  "
            f"{sp:5.2f}x │"
        )
    lines.append(f"  └{'─' * 66}┘")
    return "\n".join(lines)


def save_json(hf_stats, ll_stats, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    payload = {
        "environment": {"gpu": gpu, "cuda": torch.version.cuda, "timestamp": ts},
        "config": config,
        "results": {
            "hf": hf_stats,
            "llaisys": ll_stats,
            "speedup": {
                "wall_clock": hf_stats["elapsed_s"] / ll_stats["elapsed_s"]
                if ll_stats["elapsed_s"] > 0 else 0,
                "tpot": hf_stats["tpot_ms"] / ll_stats["tpot_ms"]
                if ll_stats["tpot_ms"] > 0 else 0,
            }
        }
    }
    path = os.path.join(run_dir, "results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON saved: {path}")
    return path
```

- [ ] **Step 2: Verify imports**

```bash
python -c "import ast; ast.parse(open('test/benchmark/infer_utils.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/infer_utils.py
git commit -m "feat: add infer_utils.py — shared inference benchmark helpers"
```

---

### Task 2: Rewrite benchmark_infer.py — single-request serial

**Files:**
- Modify: `test/benchmark/benchmark_infer.py` (complete rewrite)

- [ ] **Step 1: Write benchmark_infer.py**

```python
#!/usr/bin/env python3
"""Single-request serial inference benchmark — HF vs LLAISYS."""

import argparse, gc, sys, time
import torch
from infer_utils import (
    load_hf_model, load_llaisys_model, build_prompt_bank,
    hf_infer, llaisys_infer, summarize_run, print_speedup,
    format_table, save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_prompts", default=12, type=int)
    parser.add_argument("--context_repeat", default=2, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--repeat", default=5, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default="test/benchmark/results")
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0; args.top_k = 1; args.temperature = 1.0

    torch.manual_seed(args.seed)

    prompts = build_prompt_bank(args.num_prompts, args.context_repeat)
    max_steps, tp, tk, temp = args.max_steps, args.top_p, args.top_k, args.temperature

    # ── HF ───────────────────────────────────────────
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)

    for _ in range(args.warmup):
        for p in prompts:
            hf_infer(p, tokenizer, hf_model, max_steps, tp, tk, temp)

    hf_results = []
    hf_elapsed_sum = 0.0
    for _ in range(args.repeat):
        rep_results = []
        rep_start = time.perf_counter()
        for p in prompts:
            tokens, e_us = hf_infer(p, tokenizer, hf_model, max_steps, tp, tk, temp)
            rep_results.append((tokens, tokens, e_us))
        hf_elapsed_sum += time.perf_counter() - rep_start
        hf_results = rep_results

    hf_stats = summarize_run("HF", hf_elapsed_sum / args.repeat, hf_results)
    del hf_model; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ── LLAISYS ──────────────────────────────────────
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)

    for _ in range(args.warmup):
        for p in prompts:
            llaisys_infer(p, tokenizer, ll_model, max_steps, tp, tk, temp)

    ll_results = []
    ll_elapsed_sum = 0.0
    for _ in range(args.repeat):
        rep_results = []
        rep_start = time.perf_counter()
        for p in prompts:
            tokens, e_us = llaisys_infer(p, tokenizer, ll_model, max_steps, tp, tk, temp)
            rep_results.append((tokens, tokens, e_us))
        ll_elapsed_sum += time.perf_counter() - rep_start
        ll_results = rep_results

    ll_stats = summarize_run("LLAISYS", ll_elapsed_sum / args.repeat, ll_results)

    # ── Output ───────────────────────────────────────
    gpu = torch.cuda.get_device_name(0)
    print(f"\n  Model: Qwen2-1.5B  |  GPU: {gpu}  |  Prompts: {args.num_prompts}  |  Mode: single")
    print(format_table(hf_stats, ll_stats, "single"))
    print_speedup(hf_stats, ll_stats)

    if args.test:
        matched = 0
        for idx, (hf_r, ll_r) in enumerate(zip(hf_results, ll_results)):
            ok = hf_r[0] == ll_r[0]
            matched += int(ok)
            print(f"  Prompt [{idx}] {'OK' if ok else 'MISMATCH'}  hf={len(hf_r[0])}  ll={len(ll_r[0])}")
        print(f"  Correctness: {matched}/{len(prompts)} matched")
        if matched != len(prompts):
            sys.exit(1)

    save_json(hf_stats, ll_stats,
              {"mode": "single", "num_prompts": args.num_prompts,
               "max_steps": max_steps, "seed": args.seed},
              args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/benchmark_infer.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/benchmark_infer.py
git commit -m "refactor: rewrite benchmark_infer.py — serial single-prompt with CUDA Event + --test"
```

---

### Task 3: Rewrite benchmark_batch_infer.py — concurrent multi-request

**Files:**
- Modify: `test/benchmark/benchmark_batch_infer.py` (complete rewrite)

- [ ] **Step 1: Write benchmark_batch_infer.py**

```python
#!/usr/bin/env python3
"""Concurrent multi-request inference benchmark — HF batch vs LLAISYS concurrent."""

import argparse, gc, sys, time
import torch
from infer_utils import (
    load_hf_model, load_llaisys_model, build_prompt_bank,
    hf_batch_infer, llaisys_concurrent_infer, summarize_run, print_speedup,
    format_table, save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_prompts", default=12, type=int)
    parser.add_argument("--context_repeat", default=2, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--repeat", default=5, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default="test/benchmark/results")
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0; args.top_k = 1; args.temperature = 1.0

    torch.manual_seed(args.seed)

    prompts = build_prompt_bank(args.num_prompts, args.context_repeat)
    max_steps, tp, tk, temp = args.max_steps, args.top_p, args.top_k, args.temperature

    # ── HF batch ─────────────────────────────────────
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)

    for _ in range(args.warmup):
        hf_batch_infer(prompts, tokenizer, hf_model, max_steps, tp, tk, temp)

    hf_elapsed_sum = 0.0
    hf_last = None
    for _ in range(args.repeat):
        hf_last, e_us = hf_batch_infer(prompts, tokenizer, hf_model, max_steps, tp, tk, temp)
        hf_elapsed_sum += e_us / 1e6  # us → s

    hf_stats = summarize_run("HF", hf_elapsed_sum / args.repeat,
                              [(r[0], r[1], 0) for r in hf_last])
    del hf_model; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ── LLAISYS concurrent ───────────────────────────
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)

    for _ in range(args.warmup):
        llaisys_concurrent_infer(prompts, tokenizer, ll_model, max_steps, tp, tk, temp)

    ll_elapsed_sum = 0.0
    ll_last = None
    for _ in range(args.repeat):
        ll_last, e_s = llaisys_concurrent_infer(prompts, tokenizer, ll_model, max_steps, tp, tk, temp)
        ll_elapsed_sum += e_s

    ll_stats = summarize_run("LLAISYS", ll_elapsed_sum / args.repeat,
                              [(r[0], r[0], 0) for r in ll_last])

    # ── Output ───────────────────────────────────────
    gpu = torch.cuda.get_device_name(0)
    print(f"\n  Model: Qwen2-1.5B  |  GPU: {gpu}  |  Prompts: {args.num_prompts}  |  Mode: concurrent")
    print(format_table(hf_stats, ll_stats, "concurrent"))
    print_speedup(hf_stats, ll_stats)

    if args.test:
        matched = 0
        for idx, (hf_r, ll_r) in enumerate(zip(hf_last, ll_last)):
            ok = hf_r[1] == ll_r[0]
            matched += int(ok)
            print(f"  Prompt [{idx}] {'OK' if ok else 'MISMATCH'}  hf={len(hf_r[1])}  ll={len(ll_r[0])}")
        print(f"  Correctness: {matched}/{len(prompts)} matched")
        if matched != len(prompts):
            sys.exit(1)

    save_json(hf_stats, ll_stats,
              {"mode": "concurrent", "num_prompts": args.num_prompts,
               "max_steps": max_steps, "seed": args.seed},
              args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/benchmark_batch_infer.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/benchmark_batch_infer.py
git commit -m "refactor: rewrite benchmark_batch_infer.py — concurrent with HF batch + LLAISYS threading"
```

---

### Task 4: Final verification

- [ ] **Step 1: All files parse**

```bash
for f in test/benchmark/infer_utils.py test/benchmark/benchmark_infer.py test/benchmark/benchmark_batch_infer.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 2: Run single benchmark**

```bash
python test/benchmark/benchmark_infer.py --model <path> --warmup 1 --repeat 1 --num_prompts 4
```
Expected: table output with HF vs LLAISYS metrics, non-zero speedup.

- [ ] **Step 3: Run with --test**

```bash
python test/benchmark/benchmark_infer.py --model <path> --test --warmup 1 --repeat 1 --num_prompts 4
```
Expected: token-level comparison, all prompts matched.

- [ ] **Step 4: Run concurrent benchmark**

```bash
python test/benchmark/benchmark_batch_infer.py --model <path> --warmup 1 --repeat 1 --num_prompts 4
```
Expected: table output with HF(batch) vs LLAISYS(concurrent) metrics.
