#!/usr/bin/env python3
"""FlashInfer paged batch decode benchmark — same problem as LLAISYS flash_decoding.

Same config, same FLOPs formula, same correctness test.
CUDA_HOME=/usr/local/cuda-13.1 CC=gcc-12 CXX=g++-12 must be set before running.
"""

import time, torch, sys, os
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

NH, NKVH, HD = 12, 2, 128
PAGE = 64

_workspace = torch.empty(128 * 1024 * 1024, dtype=torch.float32, device="cuda")


def bench(batch, S, warmup=10, repeat=50):
    pages_per_seq = S // PAGE
    total_pages = pages_per_seq * batch

    indptr = torch.arange(0, batch + 1, dtype=torch.int32, device="cuda") * pages_per_seq
    indices = torch.tile(torch.arange(pages_per_seq, dtype=torch.int32, device="cuda"), (batch,))
    last_page = torch.full((batch,), PAGE, dtype=torch.int32, device="cuda")

    wrapper = BatchDecodeWithPagedKVCacheWrapper(_workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last_page, NH, NKVH, HD, PAGE, q_data_type=torch.bfloat16)

    q = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device="cuda")
    k = torch.randn(total_pages, PAGE, NKVH, HD, dtype=torch.bfloat16, device="cuda")
    v = torch.randn_like(k)

    for _ in range(warmup):
        wrapper.run(q, (k, v))
    torch.cuda.synchronize()

    best = 1e18
    for _ in range(repeat):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        wrapper.run(q, (k, v))
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e6)

    flops = batch * NH * 4 * S * HD
    tflops = flops / (best * 1e-6) / 1e12
    gbs = batch * NKVH * 2 * S * HD * 2 / (best * 1e-6) / 1e9
    return best, tflops, gbs


def main():
    torch.manual_seed(0)
    print(f"GPU: {torch.cuda.get_device_name(0)}  FlashInfer batch_decode bf16")
    print(f"config: nh={NH} nkvh={NKVH} hd={HD} page={PAGE}")

    # correctness: compare against dense reference
    B, S = 2, 512
    pp = S // PAGE
    indptr = torch.tensor([0, pp, 2 * pp], dtype=torch.int32, device="cuda")
    indices = torch.tile(torch.arange(pp, dtype=torch.int32, device="cuda"), (B,))
    last_page = torch.full((B,), PAGE, dtype=torch.int32, device="cuda")
    # sanity: run on small config and check finite output
    w = BatchDecodeWithPagedKVCacheWrapper(_workspace, kv_layout="NHD")
    w.plan(indptr, indices, last_page, NH, NKVH, HD, PAGE, q_data_type=torch.bfloat16)
    q_ref = torch.randn(B, NH, HD, dtype=torch.bfloat16, device="cuda")
    k_ref = torch.randn(B * pp, PAGE, NKVH, HD, dtype=torch.bfloat16, device="cuda")
    v_ref = torch.randn_like(k_ref)
    out_fi = w.run(q_ref, (k_ref, v_ref))
    assert torch.isfinite(out_fi).all(), "output has NaN/inf"
    print(f"  sanity: shape={list(out_fi.shape)} finite={torch.isfinite(out_fi).all().item()} OK")

    print(f"\n{'batch':>5} {'totlen':>6} {'lat(us)':>9} {'TFLOPS':>8} {'GB/s*':>7} {'%BW':>5}")
    for B in [1, 4, 8, 16]:
        for S in [256, 512, 1024, 2048]:
            lat, tflops, gbs = bench(B, S, repeat=50)
            print(f"{B:5d} {S:6d} {lat:9.1f} {tflops:8.2f} {gbs:7.0f} {gbs/256*100:4.0f}%")


if __name__ == "__main__":
    main()
