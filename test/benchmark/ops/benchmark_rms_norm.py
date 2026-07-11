#!/usr/bin/env python3
"""Benchmark: RMS normalization operator."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "rms_norm"
QWEN2_SHAPES = [
    {"shape": (1, 3584)},
    {"shape": (16, 3584)},
    {"shape": (64, 3584)},
    {"shape": (128, 3584)},
    {"shape": (512, 3584)},
    {"shape": (2048, 3584)},
]


def setup_rms_norm(shape: tuple, dtype_name: str) -> dict:
    """Allocate input, weight, and output tensors.

    inp    : *shape*
    weight : (shape[1],)
    out    : *shape*
    """
    inp_torch, inp = random_tensor(shape, dtype_name, "nvidia")
    w_torch, w = random_tensor((shape[1],), dtype_name, "nvidia")
    out_torch, out = random_tensor(shape, dtype_name, "nvidia")
    return {
        "in_torch": inp_torch, "in": inp,
        "w_torch": w_torch, "w": w,
        "out_torch": out_torch, "out": out,
    }


def torch_rms_norm(buf: dict) -> None:
    """Reference: manual PyTorch RMS normalization.

    out = x * w / sqrt(mean(x^2) + eps)
    """
    x = buf["in_torch"]
    w = buf["w_torch"]
    out = buf["out_torch"]
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    out.copy_(x * w / rms)


def llaisys_rms_norm(buf: dict) -> None:
    """LLAISYS RMS normalization operator."""
    llaisys.Ops.rms_norm(buf["out"], buf["in"], buf["w"], 1e-6)


def main():
    import argparse, os as _os
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
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
        total_flops = N * 5                      # sq + mean + rsqrt + 2*mul
        total_bytes_read = 2 * N * esize          # inp + weight (approx)
        total_bytes_write = N * esize             # out

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape={"rows": shape[0], "cols": shape[1]},
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(s=shape, d=dtype_name): return setup_rms_norm(s, d)
        def llsys_fn(buf): llaisys_rms_norm(buf)
        def ref_fn(buf): torch_rms_norm(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch_rms"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_rms_norm(shape, dtype_name)
            torch_rms_norm(buf); torch_ref = buf["out_torch"].clone()
            llaisys_rms_norm(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "RMS norm correctness check failed!"


    save_results(results, args.output)


if __name__ == "__main__":
    main()
