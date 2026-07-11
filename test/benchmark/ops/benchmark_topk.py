#!/usr/bin/env python3
"""Benchmark: top-k selection operator (values + indices along last dim)."""

import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, zero_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "top_k"
QWEN2_SHAPES = [
    {"shape": (3584,),      "k": 10},
    {"shape": (18944,),     "k": 10},
    {"shape": (1, 32000),   "k": 10},
    {"shape": (16, 32000),  "k": 10},
    {"shape": (64, 32000),  "k": 10},
    {"shape": (128, 32000), "k": 10},
]


def setup_topk(shape: tuple, k: int, dtype_name: str) -> dict:
    """Allocate input values and output index/value tensors.

    vals     : shape
    out_idx  : (shape[0], k) if 2D, else (k,) if 1D  (dtype i64)
    out_val  : (shape[0], k) if 2D, else (k,) if 1D
    """
    vals_torch, vals = random_tensor(shape, dtype_name, "nvidia")

    if len(shape) == 1:
        out_shape_idx = (k,)
        out_shape_val = (k,)
    else:
        out_shape_idx = (shape[0], k)
        out_shape_val = (shape[0], k)

    out_idx_torch, out_idx = zero_tensor(out_shape_idx, "i64", "nvidia")
    out_val_torch, out_val = zero_tensor(out_shape_val, dtype_name, "nvidia")
    return {
        "vals_torch": vals_torch, "vals": vals,
        "out_idx_torch": out_idx_torch, "out_idx": out_idx,
        "out_val_torch": out_val_torch, "out_val": out_val,
        "k": k,
    }


def torch_topk(buf: dict) -> None:
    """Reference: torch.topk, copy results to output tensors."""
    vals = buf["vals_torch"]
    out_idx = buf["out_idx_torch"]
    out_val = buf["out_val_torch"]
    k = buf["k"]

    ref_val, ref_idx = torch.topk(vals, k, dim=-1, largest=True, sorted=True)
    # Use copy_ in case shapes differ (e.g. torch may return contiguous layout)
    out_val.copy_(ref_val.contiguous().view_as(out_val))
    out_idx.copy_(ref_idx.contiguous().view_as(out_idx))


def llaisys_topk(buf: dict) -> None:
    llaisys.Ops.topk(buf["out_idx"], buf["out_val"], buf["vals"], buf["k"])


def main():
    import argparse, os as _os
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
    parser.add_argument("--ncu-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    dtype_name = args.dtype
    results = []
    for shape_info in QWEN2_SHAPES:
        shape = shape_info["shape"]
        k = shape_info["k"]
        esize = elem_size(dtype_name)

        # total elements in input
        N = 1
        for d in shape:
            N *= d

        total_bytes_read = N * esize
        # output: k values (esize) + k indices (8 bytes each for i64)
        if len(shape) == 1:
            total_bytes_write = k * esize + k * 8
        else:
            total_bytes_write = shape[0] * k * esize + shape[0] * k * 8

        # Approximate FLOPs: selection ~ N * log(k)
        total_flops = int(N * math.log(k + 1))

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape={"shape": list(shape), "k": k},
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(s=shape, kval=k, d=dtype_name):
            return setup_topk(s, kval, d)
        def llsys_fn(buf): llaisys_topk(buf)
        def ref_fn(buf): torch_topk(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch_topk"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_topk(shape, k, dtype_name)
            torch_topk(buf)
            torch_val_ref = buf["out_val_torch"].clone()
            torch_idx_ref = buf["out_idx_torch"].clone()
            llaisys_topk(buf)
            ok_val = check_equal(buf["out_val"], torch_val_ref, atol=1e-2, rtol=1e-2)
            ok_idx = check_equal(buf["out_idx"], torch_idx_ref, atol=1e-2, rtol=1e-2)
            ok = ok_val and ok_idx
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "TopK correctness check failed!"
        print(format_summary(result))

    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(set=set_map[args.ncu])
        report = profile_with_ncu(__file__,
            ["--dtype", dtype_name, "--warmup", "5", "--repeat", "1"],
            ncu_config, _os.path.join(args.output, "ncu"))
        results[0].ncu_set = report.ncu_set
        results[0].ncu_report_file = report.report_file
        results[0].ncu_bottleneck = report.bottleneck
        results[0].ncu_sm_throughput_pct = report.sm_throughput_pct
        results[0].ncu_dram_throughput_pct = report.dram_throughput_pct
        results[0].ncu_occupancy_pct = report.occupancy_pct

    if args.ncu_child:
        buf = setup_topk(QWEN2_SHAPES[0]["shape"], QWEN2_SHAPES[0]["k"], dtype_name)
        llaisys_topk(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
