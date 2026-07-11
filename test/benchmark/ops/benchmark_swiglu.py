#!/usr/bin/env python3
"""Benchmark: SwiGLU activation operator."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "swiglu"
QWEN2_SHAPES = [
    {"shape": (1, 18944)},
    {"shape": (16, 18944)},
    {"shape": (64, 18944)},
    {"shape": (128, 18944)},
    {"shape": (512, 18944)},
    {"shape": (2048, 18944)},
]


def setup_swiglu(shape: tuple, dtype_name: str) -> dict:
    """Allocate gate, up, and output tensors (all same shape)."""
    gate_torch, gate = random_tensor(shape, dtype_name, "nvidia")
    up_torch, up = random_tensor(shape, dtype_name, "nvidia")
    out_torch, out = random_tensor(shape, dtype_name, "nvidia")
    return {
        "gate_torch": gate_torch, "gate": gate,
        "up_torch": up_torch, "up": up,
        "out_torch": out_torch, "out": out,
    }


def torch_swiglu(buf: dict) -> None:
    """Reference: PyTorch SwiGLU = silu(gate) * up."""
    buf["out_torch"].copy_(F.silu(buf["gate_torch"]) * buf["up_torch"])


def llaisys_swiglu(buf: dict) -> None:
    """LLAISYS SwiGLU operator."""
    llaisys.Ops.swiglu(buf["out"], buf["gate"], buf["up"])


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
        N = shape[0] * shape[1]
        esize = elem_size(dtype_name)
        total_flops = N * 5                        # silu(~3 ops) + multiply(~1 op)
        total_bytes_read = 2 * N * esize            # gate + up
        total_bytes_write = N * esize               # out

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape={"rows": shape[0], "cols": shape[1]},
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(s=shape, d=dtype_name): return setup_swiglu(s, d)
        def llsys_fn(buf): llaisys_swiglu(buf)
        def ref_fn(buf): torch_swiglu(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch_swiglu"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_swiglu(shape, dtype_name)
            torch_swiglu(buf); torch_ref = buf["out_torch"].clone()
            llaisys_swiglu(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "SwiGLU correctness check failed!"
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
        buf = setup_swiglu(QWEN2_SHAPES[0]["shape"], dtype_name)
        llaisys_swiglu(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
