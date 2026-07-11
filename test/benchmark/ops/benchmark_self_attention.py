#!/usr/bin/env python3
"""Benchmark: self-attention operator (GQA support, causal mask)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import torch.nn.functional as F
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "self_attention"
SCALE = 1.0 / (128 ** 0.5)
QWEN2_SHAPES = [
    {"seqlen": 1,    "total_len": 64,   "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 1,    "total_len": 512,  "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 16,   "total_len": 128,  "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 64,   "total_len": 128,  "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 128,  "total_len": 256,  "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 512,  "total_len": 1024, "nh": 28, "nkvh": 4, "hd": 128},
]


def setup_self_attention(seqlen: int, total_len: int, nh: int, nkvh: int, hd: int,
                         dtype_name: str) -> dict:
    """Allocate Q/K/V and output tensors.

    q  : (seqlen, nh, hd)
    k  : (total_len, nkvh, hd)
    v  : (total_len, nkvh, hd)
    out: (seqlen, nh, hd)
    """
    q_torch, q = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    k_torch, k = random_tensor((total_len, nkvh, hd), dtype_name, "nvidia")
    v_torch, v = random_tensor((total_len, nkvh, hd), dtype_name, "nvidia")
    out_torch, out = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    return {
        "q_torch": q_torch, "q": q,
        "k_torch": k_torch, "k": k,
        "v_torch": v_torch, "v": v,
        "out_torch": out_torch, "out": out,
    }


def torch_self_attention(buf: dict) -> None:
    """Reference: F.scaled_dot_product_attention with GQA (repeat KV heads).

    Transposes to (head, seq, d) for torch's batch-first convention then back.
    is_causal=True correctly extends to rectangular masks when K/V is longer
    than Q (prefix case).
    """
    q = buf["q_torch"]                     # (seqlen, nh, hd)
    k = buf["k_torch"]                     # (total_len, nkvh, hd)
    v = buf["v_torch"]                     # (total_len, nkvh, hd)
    out = buf["out_torch"]                 # (seqlen, nh, hd)

    nh = q.shape[1]
    nkvh = k.shape[1]

    if nh != nkvh:
        repeat = nh // nkvh
        k = k.repeat_interleave(repeat, dim=1)   # (total_len, nh, hd)
        v = v.repeat_interleave(repeat, dim=1)

    # torch expects (batch, head, seq, d) — use (head, seq, d) with no batch
    q_t = q.transpose(0, 1)                # (nh, seqlen, hd)
    k_t = k.transpose(0, 1)                # (nh, total_len, hd)
    v_t = v.transpose(0, 1)                # (nh, total_len, hd)

    attn_out = F.scaled_dot_product_attention(
        q_t, k_t, v_t, is_causal=True, scale=SCALE,
    )                                      # (nh, seqlen, hd)
    out.copy_(attn_out.transpose(0, 1))    # (seqlen, nh, hd)


def llaisys_self_attention(buf: dict) -> None:
    llaisys.Ops.self_attention(buf["out"], buf["q"], buf["k"], buf["v"], SCALE)


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
        seqlen = shape_info["seqlen"]
        total_len = shape_info["total_len"]
        nh = shape_info["nh"]
        nkvh = shape_info["nkvh"]
        hd = shape_info["hd"]
        esize = elem_size(dtype_name)

        N_q = seqlen * nh * hd
        N_kv = total_len * nkvh * hd * 2           # k + v
        total_bytes_read = (N_q + N_kv) * esize
        total_bytes_write = N_q * esize
        # 2*QK^T + 2*PV  = 4 * seqlen * total_len * nh * hd
        total_flops = 4 * seqlen * total_len * nh * hd

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape=shape_info,
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(sl=seqlen, tl=total_len, n=nh, nk=nkvh, h=hd, d=dtype_name):
            return setup_self_attention(sl, tl, n, nk, h, d)
        def llsys_fn(buf): llaisys_self_attention(buf)
        def ref_fn(buf): torch_self_attention(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch_flash_attn"
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_self_attention(seqlen, total_len, nh, nkvh, hd, dtype_name)
            torch_self_attention(buf); torch_ref = buf["out_torch"].clone()
            llaisys_self_attention(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Self-attention correctness check failed!"
        print(format_summary(result))


    save_results(results, args.output)


if __name__ == "__main__":
    main()
