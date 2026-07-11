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
    parser.add_argument("--ncu-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    dtype_name = args.dtype
    results = []
    for shape_info in QWEN2_SHAPES:
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
        print(format_summary(result))

    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(set=set_map[args.ncu], kernel_name=f".*{OP_NAME}.*")
        report = profile_with_ncu(__file__, ["--dtype", dtype_name, "--warmup", "5", "--repeat", "1"],
            ncu_config, _os.path.join(args.output, "ncu"))
        results[0].ncu_set = report.ncu_set
        results[0].ncu_report_file = report.report_file
        results[0].ncu_bottleneck = report.bottleneck
        results[0].ncu_sm_throughput_pct = report.sm_throughput_pct
        results[0].ncu_dram_throughput_pct = report.dram_throughput_pct
        results[0].ncu_occupancy_pct = report.occupancy_pct

    if args.ncu_child:
        buf = setup_argmax(QWEN2_SHAPES[0]["shape"], dtype_name)
        llaisys_argmax(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)

if __name__ == "__main__":
    main()
