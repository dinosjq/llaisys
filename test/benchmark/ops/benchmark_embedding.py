#!/usr/bin/env python3
"""Benchmark: embedding lookup operator."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, random_int_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "embedding"
QWEN2_SHAPES = [
    {"seqlen": 1, "vocab": 151936, "dim": 3584},
    {"seqlen": 16, "vocab": 151936, "dim": 3584},
    {"seqlen": 64, "vocab": 151936, "dim": 3584},
    {"seqlen": 128, "vocab": 151936, "dim": 3584},
    {"seqlen": 512, "vocab": 151936, "dim": 3584},
    {"seqlen": 2048, "vocab": 151936, "dim": 3584},
]

def torch_embedding(buffers: dict) -> None:
    buffers["out_torch"].copy_(torch.nn.functional.embedding(buffers["idx_torch"], buffers["w_torch"]))

def llaisys_embedding(buffers: dict) -> None:
    llaisys.Ops.embedding(buffers["out"], buffers["idx"], buffers["w"])

def setup_embedding(shape_info: dict, dtype_name: str) -> dict:
    seqlen = shape_info["seqlen"]
    vocab = shape_info["vocab"]
    dim = shape_info["dim"]
    idx_torch, idx = random_int_tensor((seqlen,), "nvidia", "i64", high=vocab)
    w_torch, w = random_tensor((vocab, dim), dtype_name, "nvidia")
    out_torch, out = random_tensor((seqlen, dim), dtype_name, "nvidia")
    return {"idx_torch": idx_torch, "idx": idx,
            "w_torch": w_torch, "w": w,
            "out_torch": out_torch, "out": out}

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
        seqlen = shape_info["seqlen"]
        dim = shape_info["dim"]
        esize = elem_size(dtype_name)
        config = BenchmarkConfig(op_name=OP_NAME, shape=shape_info,
            dtype_name=dtype_name, warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=(seqlen*8 + seqlen*dim*esize),
            total_bytes_write=seqlen*dim*esize, total_flops=0)

        def setup_fn(s=shape_info, d=dtype_name): return setup_embedding(s, d)
        def llsys_fn(buf): llaisys_embedding(buf)
        def ref_fn(buf): torch_embedding(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_embedding(shape_info, dtype_name)
            torch_embedding(buf); torch_ref = buf["out_torch"].clone()
            llaisys_embedding(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Embedding correctness check failed!"
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
        buf = setup_embedding(QWEN2_SHAPES[0], dtype_name)
        llaisys_embedding(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)

if __name__ == "__main__":
    main()
