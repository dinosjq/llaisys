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
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--variant", default="all",
                        choices=["all", "q_proj", "gate_proj", "down_proj"])
    parser.add_argument("--M", type=int, default=0,
                        help="M dimension (0 = all M_VALUES)")
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
    parser.add_argument("--ncu-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    dtype_name = args.dtype

    # Determine variants to run
    if args.variant == "all":
        variants = list(QWEN2_VARIANTS.items())
    else:
        variants = [(args.variant, QWEN2_VARIANTS[args.variant])]

    # Determine M values to run
    ms = M_VALUES if args.M == 0 else [args.M]

    results = []
    for variant_name, (N, K) in variants:
        for M in ms:
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
            print(format_summary(result))

    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(set=set_map[args.ncu], kernel_name=f".*{OP_NAME}.*")
        report = profile_with_ncu(__file__,
            ["--dtype", dtype_name, "--warmup", "5", "--repeat", "1",
             "--variant", "q_proj", "--M", "128"],
            ncu_config, _os.path.join(args.output, "ncu"))
        results[0].ncu_set = report.ncu_set
        results[0].ncu_report_file = report.report_file
        results[0].ncu_bottleneck = report.bottleneck
        results[0].ncu_sm_throughput_pct = report.sm_throughput_pct
        results[0].ncu_dram_throughput_pct = report.dram_throughput_pct
        results[0].ncu_occupancy_pct = report.occupancy_pct

    if args.ncu_child:
        # NCU profiles q_proj M=128
        buf = setup_linear(M=128, N=3584, K=3584, dtype=dtype_name)
        llaisys_linear(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
