#!/usr/bin/env python3
"""W8A16 performance: linear microbench + optional e2e FP vs W8 load/generate."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

import llaisys
from test_utils import llaisys_device, random_tensor, torch_device
from transformers import AutoTokenizer

from infer_utils import (
    build_input_ids,
    load_llaisys_model,
    llaisys_generate_serial,
)

# Shapes: (name, M, N, K) — weight is [N, K], input [M, K], out [M, N]
# Qwen2 1.5B: H=1536, I=8960, nh=12, nkv=2, hd=128 → q=1536, kv=256
QWEN15B_SHAPES = [
    ("qwen_q_decode", 1, 1536, 1536),
    ("qwen_k_decode", 1, 256, 1536),
    ("qwen_o_decode", 1, 1536, 1536),
    ("qwen_gate_decode", 1, 8960, 1536),
    ("qwen_down_decode", 1, 1536, 8960),
    ("qwen_q_m8", 8, 1536, 1536),
    ("qwen_gate_m8", 8, 8960, 1536),
    ("qwen_q_prefill128", 128, 1536, 1536),
    ("qwen_gate_prefill128", 128, 8960, 1536),
]

# Llama-3.2-3B: H=3072, I=8192, nh=24, nkv=8, hd=128 → q=3072, kv=1024
LLAMA3B_SHAPES = [
    ("llama_q_decode", 1, 3072, 3072),
    ("llama_k_decode", 1, 1024, 3072),
    ("llama_o_decode", 1, 3072, 3072),
    ("llama_gate_decode", 1, 8192, 3072),
    ("llama_down_decode", 1, 3072, 8192),
    ("llama_q_m8", 8, 3072, 3072),
    ("llama_gate_m8", 8, 8192, 3072),
    ("llama_q_prefill128", 128, 3072, 3072),
    ("llama_gate_prefill128", 128, 8192, 3072),
]


def _sync(device_name: str):
    if device_name == "nvidia":
        torch.cuda.synchronize()


def _time_cuda_ms(fn, warmup: int, repeat: int, device_name: str) -> float:
    for _ in range(warmup):
        fn()
    _sync(device_name)
    if device_name == "nvidia":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeat):
            fn()
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / repeat
    t0 = time.perf_counter()
    for _ in range(repeat):
        fn()
    return (time.perf_counter() - t0) * 1000.0 / repeat


def bench_linear_pair(
    m: int,
    n: int,
    k: int,
    dtype_name: str,
    device_name: str,
    warmup: int,
    repeat: int,
) -> dict:
    x, x_ = random_tensor((m, k), dtype_name, device_name, scale=0.1)
    w_fp, w_fp_ = random_tensor((n, k), dtype_name, device_name, scale=0.05)
    _, out_fp_ = random_tensor((m, n), dtype_name, device_name)
    _, out_w8_ = random_tensor((m, n), dtype_name, device_name)

    w_i8 = llaisys.Tensor(
        [n, k],
        dtype=llaisys.DataType.I8,
        device=llaisys_device(device_name),
        device_id=0,
    )
    scale_t = llaisys.Tensor(
        [n],
        dtype=llaisys.DataType.F32,
        device=llaisys_device(device_name),
        device_id=0,
    )
    llaisys.Ops.quantize_w8(w_i8, scale_t, w_fp_)
    _sync(device_name)

    t_fp = _time_cuda_ms(
        lambda: llaisys.Ops.linear(out_fp_, x_, w_fp_, None),
        warmup,
        repeat,
        device_name,
    )
    t_w8 = _time_cuda_ms(
        lambda: llaisys.Ops.linear_w8a16(out_w8_, x_, w_i8, scale_t, None),
        warmup,
        repeat,
        device_name,
    )
    t_quant = _time_cuda_ms(
        lambda: llaisys.Ops.quantize_w8(w_i8, scale_t, w_fp_),
        max(2, warmup // 2),
        max(10, repeat // 5),
        device_name,
    )
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": dtype_name,
        "linear_fp_ms": round(t_fp, 5),
        "linear_w8a16_ms": round(t_w8, 5),
        "speedup_fp_over_w8": round(t_fp / t_w8, 3) if t_w8 > 0 else None,
        "quantize_w8_ms": round(t_quant, 5),
        "weight_fp_bytes": n * k * (2 if dtype_name == "bf16" else 4),
        "weight_w8_bytes": n * k + n * 4,
    }


def run_microbench(args) -> list[dict]:
    shapes = []
    if args.shapes in ("qwen", "all"):
        shapes.extend(QWEN15B_SHAPES)
    if args.shapes in ("llama", "all"):
        shapes.extend(LLAMA3B_SHAPES)

    rows = []
    for name, m, n, k in shapes:
        row = bench_linear_pair(
            m, n, k, args.dtype, args.device, args.op_warmup, args.op_repeat
        )
        row["name"] = name
        rows.append(row)
        print(
            f"  {name:28s} M={m:4d} N={n:5d} K={k:5d}  "
            f"fp={row['linear_fp_ms']:.4f} ms  w8={row['linear_w8a16_ms']:.4f} ms  "
            f"x{row['speedup_fp_over_w8']:.2f}"
        )
    return rows


def _peak_mem_mib() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_e2e(args) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cases = []
    for i, length in enumerate(args.input_lens):
        ids = build_input_ids(tokenizer, length, seed_offset=i)
        cases.append({"input_ids": ids, "input_len": length, "case_id": i})

    results = {"fp": {}, "w8": {}}
    for label, quant in (("fp", False), ("w8", True)):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        t_load0 = time.perf_counter()
        model = load_llaisys_model(args.model, args.device, quantize_weights=quant)
        torch.cuda.synchronize()
        load_s = time.perf_counter() - t_load0
        peak_after_load = _peak_mem_mib()

        # warmup
        for _ in range(args.warmup):
            llaisys_generate_serial(
                cases[:1],
                model,
                args.max_steps,
                args.top_p,
                args.top_k,
                args.temperature,
            )

        walls = []
        out_tokens_total = 0
        for _ in range(args.repeat):
            gen_results, wall = llaisys_generate_serial(
                cases,
                model,
                args.max_steps,
                args.top_p,
                args.top_k,
                args.temperature,
            )
            walls.append(wall)
            out_tokens_total = sum(
                max(0, len(full) - case["input_len"])
                for (full, _, _), case in zip(gen_results, cases)
            )

        best_wall = min(walls)
        tok_s = out_tokens_total / best_wall if best_wall > 0 else 0.0
        peak = _peak_mem_mib()
        results[label] = {
            "quantize_weights": quant,
            "load_s": round(load_s, 3),
            "peak_mem_mib_after_load": round(peak_after_load, 1) if peak_after_load else None,
            "peak_mem_mib": round(peak, 1) if peak else None,
            "best_wall_s": round(best_wall, 4),
            "output_tokens": out_tokens_total,
            "tok_per_s": round(tok_s, 2),
            "walls_s": [round(w, 4) for w in walls],
        }
        print(
            f"  e2e {label}: load={load_s:.2f}s  tok/s={tok_s:.2f}  "
            f"peak={peak:.0f} MiB  wall={best_wall:.3f}s"
        )
        # drop model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fp = results["fp"]
    w8 = results["w8"]
    results["speedup_w8_over_fp"] = (
        round(w8["tok_per_s"] / fp["tok_per_s"], 3)
        if fp.get("tok_per_s")
        else None
    )
    results["load_ratio_w8_over_fp"] = (
        round(w8["load_s"] / fp["load_s"], 3) if fp.get("load_s") else None
    )
    return results


def main():
    p = argparse.ArgumentParser(description="W8A16 performance benchmark")
    p.add_argument("--device", default="nvidia")
    p.add_argument("--dtype", default="bf16", choices=("bf16", "f32"))
    p.add_argument("--shapes", default="all", choices=("qwen", "llama", "all"))
    p.add_argument("--op-warmup", type=int, default=20)
    p.add_argument("--op-repeat", type=int, default=100)
    p.add_argument("--skip-micro", action="store_true")
    p.add_argument("--skip-e2e", action="store_true")
    p.add_argument(
        "--model",
        default="/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B",
    )
    p.add_argument("--input-lens", default="32,128", type=str)
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=2)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument(
        "--output",
        default="",
        help="Directory for results.json (default under test/results/)",
    )
    args = p.parse_args()
    args.input_lens = [int(x) for x in args.input_lens.split(",") if x.strip()]

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output) if args.output else Path(
        f"test/results/w8a16_bench_{stamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    payload = {
        "timestamp_utc": stamp,
        "device": args.device,
        "dtype": args.dtype,
        "gpu": gpu_name,
        "microbench": [],
        "e2e": None,
    }

    print("=== W8A16 microbench (linear vs linear_w8a16) ===")
    if not args.skip_micro:
        payload["microbench"] = run_microbench(args)
    else:
        print("  skipped")

    print("=== W8A16 e2e (FP vs quantize_weights) ===")
    if not args.skip_e2e:
        payload["e2e"] = {
            "model": args.model,
            "input_lens": args.input_lens,
            "max_steps": args.max_steps,
            "warmup": args.warmup,
            "repeat": args.repeat,
            **run_e2e(args),
        }
    else:
        print("  skipped")

    out_path = out_dir / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
