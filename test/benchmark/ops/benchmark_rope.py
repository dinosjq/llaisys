#!/usr/bin/env python3
"""Benchmark: Rotary Position Embedding (RoPE) operator."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, random_int_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "rope"
THETA = 10000.0
QWEN2_SHAPES = [
    {"seqlen": 1,    "nhead": 28, "hd": 128},
    {"seqlen": 1,    "nhead": 4,  "hd": 128},
    {"seqlen": 16,   "nhead": 28, "hd": 128},
    {"seqlen": 128,  "nhead": 28, "hd": 128},
    {"seqlen": 512,  "nhead": 28, "hd": 128},
    {"seqlen": 2048, "nhead": 28, "hd": 128},
]


def setup_rope(seqlen: int, nhead: int, hd: int, dtype_name: str) -> dict:
    """Allocate input, position ids, and output tensors.

    inp     : (seqlen, nhead, hd)
    pos_ids : (seqlen,)  i64, random in [0, 2048)
    out     : (seqlen, nhead, hd)
    """
    inp_torch, inp = random_tensor((seqlen, nhead, hd), dtype_name, "nvidia")
    pos_torch, pos = random_int_tensor((seqlen,), "nvidia", dtype_name="i64",
                                       low=0, high=2048)
    out_torch, out = random_tensor((seqlen, nhead, hd), dtype_name, "nvidia")
    return {
        "in_torch": inp_torch, "in": inp,
        "pos_torch": pos_torch, "pos": pos,
        "out_torch": out_torch, "out": out,
    }


def torch_rope(buf: dict) -> None:
    """Reference: manual RoPE matching the LLAISYS half-split layout.

    LLAISYS splits the last dimension into two halves (not interleaved pairs).
    For position p and index j in [0, hd/2):
        den = THETA^(2j / hd)
        phi = p / den         (equivalent to p * THETA^(-2j/hd))
        cos_j = cos(phi),  sin_j = sin(phi)
        in_a = inp[..., j]           (first half)
        in_b = inp[..., j + hd/2]    (second half)
        out_a = in_a*cos_j - in_b*sin_j
        out_b = in_b*cos_j + in_a*sin_j
    """
    inp = buf["in_torch"]           # (seqlen, nhead, hd)
    pos = buf["pos_torch"]          # (seqlen,)  i64
    out = buf["out_torch"]          # (seqlen, nhead, hd)

    seqlen, nhead, hd = inp.shape
    d2 = hd // 2
    dtype = inp.dtype
    device = inp.device

    # Frequencies: theta^(2*j/d) for j = 0..d2-1  (computed in f32 for stability)
    j_idx = torch.arange(0, d2, device=device, dtype=torch.float32)
    den = THETA ** (2.0 * j_idx / hd)              # (d2,)
    # Equivalently: freq = 1/den = THETA^(-2*j/hd)

    # Phase angles: pos (seqlen,1) / den (1,d2) -> (seqlen, d2)
    pos_f = pos.to(torch.float32)[:, None]          # (seqlen, 1)
    phi = pos_f / den[None, :]                      # (seqlen, d2)
    cos = torch.cos(phi).to(dtype)[:, None, :]      # (seqlen, 1, d2)
    sin = torch.sin(phi).to(dtype)[:, None, :]      # (seqlen, 1, d2)

    # Split last dim: first half / second half  (not even/odd interleaved)
    inp_a = inp[..., :d2]                            # (seqlen, nhead, d2) -- first half
    inp_b = inp[..., d2:]                            # (seqlen, nhead, d2) -- second half

    out_a = inp_a * cos - inp_b * sin
    out_b = inp_b * cos + inp_a * sin

    out.copy_(torch.cat([out_a, out_b], dim=-1))


def llaisys_rope(buf: dict) -> None:
    """LLAISYS RoPE operator."""
    llaisys.Ops.rope(buf["out"], buf["in"], buf["pos"], THETA)


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
        seqlen = shape_info["seqlen"]
        nhead = shape_info["nhead"]
        hd = shape_info["hd"]
        esize = elem_size(dtype_name)
        N = seqlen * nhead * hd
        total_flops = N * 4                        # ~4 ops per element for rotation
        total_bytes_read = N * esize + seqlen * 8  # inp + pos_ids
        total_bytes_write = N * esize              # out

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape=shape_info,
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(sl=seqlen, nh=nhead, h=hd, d=dtype_name):
            return setup_rope(sl, nh, h, d)
        def llsys_fn(buf): llaisys_rope(buf)
        def ref_fn(buf): torch_rope(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch_rope"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_rope(seqlen, nhead, hd, dtype_name)
            torch_rope(buf); torch_ref = buf["out_torch"].clone()
            llaisys_rope(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "RoPE correctness check failed!"
        print(format_summary(result))

    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(set=set_map[args.ncu], kernel_name=f".*{OP_NAME}.*")
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
        buf = setup_rope(QWEN2_SHAPES[0]["seqlen"], QWEN2_SHAPES[0]["nhead"],
                         QWEN2_SHAPES[0]["hd"], dtype_name)
        llaisys_rope(buf); torch.cuda.synchronize(); return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
