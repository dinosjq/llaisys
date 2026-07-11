#!/usr/bin/env python3
"""Benchmark: argmax operator."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, random_int_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "argmax"
SHAPE_LABEL = "shape:[rows x cols]"
QWEN2_SHAPES = [
    {"shape": (1, 3584)},
    {"shape": (64, 3584)},
    {"shape": (128, 3584)},
    {"shape": (512, 3584)},
    {"shape": (2048, 3584)},
    {"shape": (2048, 18944)},
]

def torch_argmax(buffers: dict) -> None:
    torch.argmax(buffers["vals_torch"], dim=-1, out=buffers["max_idx_torch"])
    torch.max(buffers["vals_torch"], dim=-1, out=(buffers["max_val_torch"], buffers["max_idx_torch"]))

def llaisys_argmax(buffers: dict) -> None:
    llaisys.Ops.argmax(buffers["max_idx"], buffers["max_val"], buffers["vals"])

def setup_argmax(shape: tuple, dtype_name: str) -> dict:
    vals_torch, vals = random_tensor(shape, dtype_name, "nvidia")
    max_val_torch, max_val = random_tensor((shape[0],), dtype_name, "nvidia")
    max_idx_torch, max_idx = random_int_tensor((shape[0],), "nvidia", "i64")
    return {"vals_torch": vals_torch, "vals": vals,
            "max_val_torch": max_val_torch, "max_val": max_val,
            "max_idx_torch": max_idx_torch, "max_idx": max_idx}

def main():
    import argparse, os as _os
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--ncu", nargs="?", const="detailed", choices=["quick", "detailed", "full"])
    parser.add_argument("--use-ncu", action="store_true", help="Profile with NCU (--set full)")
    parser.add_argument("--example-index", type=str, default=None, help="Comma-separated shape indices to profile (0-based)")
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    target_indices = None
    if args.example_index is not None:
        target_indices = sorted(set(
            int(p.strip()) for p in args.example_index.split(",") if p.strip()))
        total = len(QWEN2_SHAPES)
        for idx in target_indices:
            if idx < 0 or idx >= total:
                print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                sys.exit(1)


    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20

    dtype_name = args.dtype
    results = []
    _idx = 0
    for shape_info in QWEN2_SHAPES:
        this_idx = _idx
        _idx += 1
        if target_indices is not None and this_idx not in target_indices:
            continue
        shape = shape_info["shape"]
        N = shape[0] * shape[1]
        esize = elem_size(dtype_name)
        config = BenchmarkConfig(op_name=OP_NAME, shape={"rows": shape[0], "cols": shape[1]},
            dtype_name=dtype_name, warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=N*esize, total_bytes_write=shape[0]*(esize+8), total_flops=0)

        def setup_fn(s=shape, d=dtype_name): return setup_argmax(s, d)
        def llsys_fn(buf): llaisys_argmax(buf)
        def ref_fn(buf): torch_argmax(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch"
        results.append(result)

        if shape == QWEN2_SHAPES[0]["shape"]:
            buf = setup_argmax(shape, dtype_name)
            torch_argmax(buf); torch_ref = buf["max_val_torch"].clone()
            llaisys_argmax(buf)
            ok = check_equal(buf["max_val"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Argmax correctness check failed!"


    save_results(results, args.output, SHAPE_LABEL)

    if args.use_ncu:
        indices = target_indices
        if indices is None:
            speeds = [(i, r.speedup) for i, r in enumerate(results)]
            below = [(i, s) for i, s in speeds if s < 1.0]
            if below:
                indices = [min(below, key=lambda x: x[1])[0]]
            else:
                indices = [min(speeds, key=lambda x: abs(x[1] - 1.0))[0]]
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)

if __name__ == "__main__":
    main()
