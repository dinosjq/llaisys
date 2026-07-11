#!/usr/bin/env python3
"""Benchmark: element-wise addition (add)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "add"
QWEN2_SHAPES = [
    {"shape": (1, 3584)},
    {"shape": (16, 3584)},
    {"shape": (64, 3584)},
    {"shape": (128, 3584)},
    {"shape": (512, 3584)},
    {"shape": (2048, 3584)},
]

def torch_add(buffers: dict) -> None:
    torch.add(buffers["a_torch"], buffers["b_torch"], out=buffers["out_torch"])

def llaisys_add(buffers: dict) -> None:
    llaisys.Ops.add(buffers["out"], buffers["a"], buffers["b"])

def setup_add(shape: tuple, dtype_name: str) -> dict:
    a_torch, a = random_tensor(shape, dtype_name, "nvidia")
    b_torch, b = random_tensor(shape, dtype_name, "nvidia")
    out_torch, out = random_tensor(shape, dtype_name, "nvidia")
    return {"a_torch": a_torch, "a": a, "b_torch": b_torch, "b": b, "out_torch": out_torch, "out": out}

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

    if args.use_ncu:
        indices = None
        if args.example_index is not None:
            indices = sorted(set(int(p.strip()) for p in args.example_index.split(",") if p.strip()))
            total = len(QWEN2_SHAPES)
            for idx in indices:
                if idx < 0 or idx >= total:
                    print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                    sys.exit(1)
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME, example_indices=indices)
        return

    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20

    dtype_name = args.dtype
    results = []
    for shape_info in QWEN2_SHAPES:
        shape = shape_info["shape"]
        N = shape[0] * shape[1]
        esize = elem_size(dtype_name)
        total_bytes = N * esize
        config = BenchmarkConfig(op_name=OP_NAME, shape={"rows": shape[0], "cols": shape[1]},
            dtype_name=dtype_name, warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=2*total_bytes, total_bytes_write=total_bytes, total_flops=0)

        def setup_fn(s=shape, d=dtype_name): return setup_add(s, d)
        def llsys_fn(buf): llaisys_add(buf)
        def ref_fn(buf): torch_add(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch"
        results.append(result)

        if shape == QWEN2_SHAPES[0]["shape"]:
            buf = setup_add(shape, dtype_name)
            torch_add(buf); torch_ref = buf["out_torch"].clone()
            llaisys_add(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Add correctness check failed!"


    save_results(results, args.output)

if __name__ == "__main__":
    main()
