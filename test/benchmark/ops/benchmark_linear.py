#!/usr/bin/env python3
"""Benchmark: linear (matrix multiply with optional bias, backed by cuBLAS)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "linear"
SHAPE_LABEL = "shape:[M x N x K x variant]"
QWEN2_VARIANTS = {
    "q_proj":    (3584, 3584),   # N, K  (Q/K/V/O proj)
    "gate_proj": (18944, 3584),  # N, K  (Gate/Up proj)
    "down_proj": (3584, 18944),  # N, K  (Down proj)
}
M_VALUES = [1, 16, 64, 128, 512, 2048]


def setup_linear(M: int, N: int, K: int, dtype_name: str, use_bias: bool = True) -> dict:
    """Allocate input, weight, output, and optional bias tensors.

    inp : (M, K)
    weight : (N, K)   -- transposed by cuBLAS; stored as (out_features, in_features)
    out : (M, N)
    bias : (N,)       -- only when *use_bias* is True
    """
    inp_torch, inp = random_tensor((M, K), dtype_name, "nvidia")
    w_torch, w = random_tensor((N, K), dtype_name, "nvidia")
    out_torch, out = random_tensor((M, N), dtype_name, "nvidia")
    buf = {
        "in_torch": inp_torch, "in": inp,
        "w_torch": w_torch, "w": w,
        "out_torch": out_torch, "out": out,
        "use_bias": use_bias,
    }
    if use_bias:
        bias_torch, bias = random_tensor((N,), dtype_name, "nvidia")
        buf["bias_torch"] = bias_torch
        buf["bias"] = bias
    else:
        buf["bias_torch"] = None
        buf["bias"] = None
    return buf


def torch_linear(buf: dict) -> None:
    """Reference: PyTorch F.linear (calls cuBLAS under the hood)."""
    F.linear(buf["in_torch"], buf["w_torch"],
             bias=buf["bias_torch"], out=buf["out_torch"])


def llaisys_linear(buf: dict) -> None:
    """LLAISYS linear operator (calls cuBLAS gemm + bias add)."""
    llaisys.Ops.linear(buf["out"], buf["in"], buf["w"],
                       buf["bias"] if buf["use_bias"] else None)


def main():
    import argparse, os as _os
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=10,
                        help="Timed iterations per shape (max 20)")
    parser.add_argument("--variant", default="all",
                        choices=["all", "q_proj", "gate_proj", "down_proj"])
    parser.add_argument("--M", type=int, default=0,
                        help="M dimension (0 = all M_VALUES)")
    parser.add_argument("--use-ncu", action="store_true",
                        help="Profile with NCU (--set full)")
    parser.add_argument("--example-index", type=str, default=None,
                        help="Comma-separated shape indices to profile (0-based)")
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    # Parse --example-index for shape filtering (used by --use-ncu subprocess)
    target_indices = None
    if args.example_index is not None:
        target_indices = sorted(set(
            int(p.strip()) for p in args.example_index.split(",") if p.strip()))
        total = len(QWEN2_VARIANTS) * len(M_VALUES)
        for idx in target_indices:
            if idx < 0 or idx >= total:
                print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                sys.exit(1)

    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20

    dtype_name = args.dtype

    # Determine variants to run
    if args.variant == "all":
        variants = list(QWEN2_VARIANTS.items())
    else:
        variants = [(args.variant, QWEN2_VARIANTS[args.variant])]

    # Determine M values to run
    ms = M_VALUES if args.M == 0 else [args.M]

    results = []
    idx_counter = 0
    for variant_name, (N, K) in variants:
        for M in ms:
            # --example-index filtering
            this_idx = idx_counter
            idx_counter += 1
            if target_indices is not None and this_idx not in target_indices:
                continue

            esize = elem_size(dtype_name)
            total_flops = 2 * M * N * K        # GEMM flops
            total_bytes_read = (M * K + N * K + N) * esize   # inp + weight + bias
            total_bytes_write = M * N * esize                  # output

            config = BenchmarkConfig(
                op_name=OP_NAME,
                shape={"M": M, "N": N, "K": K, "variant": variant_name},
                dtype_name=dtype_name,
                warmup=args.warmup, repeat=args.repeat,
                total_bytes_read=total_bytes_read,
                total_bytes_write=total_bytes_write,
                total_flops=total_flops,
            )

            def setup_fn(m=M, n=N, k=K, d=dtype_name):
                return setup_linear(m, n, k, d)
            def llsys_fn(buf): llaisys_linear(buf)
            def ref_fn(buf): torch_linear(buf)

            result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
            result.baseline_name = "cuBLAS"
            results.append(result)

            # Correctness: validate on q_proj with M=1 (when that combo runs)
            if M == 1 and variant_name == "q_proj":
                buf = setup_linear(M, N, K, dtype_name)
                torch_linear(buf); torch_ref = buf["out_torch"].clone()
                llaisys_linear(buf)
                ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
                print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
                assert ok, "Linear correctness check failed!"

    save_results(results, args.output, SHAPE_LABEL)

    if args.use_ncu:
        indices = target_indices  # may be None (auto-select) or a list
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
