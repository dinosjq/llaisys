#!/usr/bin/env python3
"""Microbench / ncu target for linear_w8a16 (decode-like M)."""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import llaisys
from test_utils import random_tensor, llaisys_device


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=1)
    p.add_argument("--n", type=int, default=1536)
    p.add_argument("--k", type=int, default=1536)
    p.add_argument("--dtype", default="bf16")
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--repeat", type=int, default=200)
    p.add_argument("--compare-fp", action="store_true")
    args = p.parse_args()

    device = "nvidia"
    m, n, k = args.m, args.n, args.k
    _, x_ = random_tensor((m, k), args.dtype, device, scale=0.1)
    w_fp, w_fp_ = random_tensor((n, k), args.dtype, device, scale=0.05)
    _, out_ = random_tensor((m, n), args.dtype, device)
    _, out_fp_ = random_tensor((m, n), args.dtype, device)

    w_i8 = llaisys.Tensor([n, k], dtype=llaisys.DataType.I8, device=llaisys_device(device), device_id=0)
    scale = llaisys.Tensor([n], dtype=llaisys.DataType.F32, device=llaisys_device(device), device_id=0)
    llaisys.Ops.quantize_w8(w_i8, scale, w_fp_)
    torch.cuda.synchronize()

    for _ in range(args.warmup):
        llaisys.Ops.linear_w8a16(out_, x_, w_i8, scale, None)
        if args.compare_fp:
            llaisys.Ops.linear(out_fp_, x_, w_fp_, None)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.repeat):
        llaisys.Ops.linear_w8a16(out_, x_, w_i8, scale, None)
    end.record()
    torch.cuda.synchronize()
    w8_ms = start.elapsed_time(end) / args.repeat

    fp_ms = None
    if args.compare_fp:
        start.record()
        for _ in range(args.repeat):
            llaisys.Ops.linear(out_fp_, x_, w_fp_, None)
        end.record()
        torch.cuda.synchronize()
        fp_ms = start.elapsed_time(end) / args.repeat

    print(f"shape m={m} n={n} k={k} dtype={args.dtype}")
    print(f"linear_w8a16: {w8_ms:.5f} ms")
    if fp_ms is not None:
        print(f"linear_fp:    {fp_ms:.5f} ms  ratio_fp/w8={fp_ms/w8_ms:.3f}")


if __name__ == "__main__":
    main()
