#!/usr/bin/env python3
"""Benchmark: flash_decoding (decode-stage paged attention), compute TFLOP.

FLOPs formula
-------------
Per (query token, head), attention over S = totlen tokens with head dim hd:
    QK^T  : 2 * S * hd   (matmul)
    P@V   : 2 * S * hd   (matmul)
    Total = 4 * S * hd FLOPs per (query, head).
    Grand total = batch * nh * 4 * totlen * hd.

Decode attention is memory-bound (K/V streaming dominates), so achieved
TFLOPS is expected to be a small fraction of the GPU's FP16 tensor peak;
the achieved memory bandwidth (GB/s) vs the GPU's peak HBM bandwidth is the
more meaningful signal.  Both are reported here.

Shapes use the Qwen2-1.5B-scale decode config from test/ops/flash_decoding.py:
    nh=14, nkvh=2, hd=dv=64, token_num=64.
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import torch
import llaisys
from ctypes import c_void_p
from test_utils import check_equal
from benchmark_harness import BenchmarkConfig, run_benchmark, save_results, elem_size

OP_NAME = "flash_decoding"
SHAPE_LABEL = "batch x totlen"

# Real model: DeepSeek-R1-Distill-Qwen-1.5B (config.json)
#   num_attention_heads=12, num_key_value_heads=2, head_dim=1536/12=128
NH = 12
NKVH = 2
HD = 128
TOKEN_NUM = 64

BATCH_SIZES = [1, 4, 8, 16]
TOTLENS = [256, 512, 1024, 2048]


def torch_flash_decoding(attn_val, query, k_cache, v_cache, block_ids_values, scale, totlen):
    """Reference decode attention (seqlen=1, attends over all totlen keys)."""
    seqlen, nh, hd = query.shape
    _, token_num, nkvh, _ = k_cache.shape

    dense_k = torch.empty((totlen, nkvh, hd), dtype=query.dtype, device=query.device)
    dense_v = torch.empty((totlen, nkvh, hd), dtype=query.dtype, device=query.device)
    for token_id in range(totlen):
        block_id = int(block_ids_values[token_id // token_num])
        local_id = token_id % token_num
        dense_k[token_id] = k_cache[block_id, local_id]
        dense_v[token_id] = v_cache[block_id, local_id]

    if nh != nkvh:
        repeat = nh // nkvh
        dense_k = dense_k.repeat_interleave(repeat, dim=1)
        dense_v = dense_v.repeat_interleave(repeat, dim=1)

    q = query.transpose(0, 1)
    k = dense_k.transpose(0, 1)
    v = dense_v.transpose(0, 1)

    attn_weight = torch.matmul(q, k.transpose(-2, -1)) * scale
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_val.copy_(torch.matmul(attn_weight, v).transpose(0, 1))


def make_buffers(batch: int, totlen: int, dtype_name: str):
    """Pre-allocate all flash_decoding tensors for a (batch, totlen) config."""
    dev_torch = torch.device("cuda:0")
    block_num = totlen // TOKEN_NUM
    mb = block_num + 1  # +1 for the prefix count column

    # block_ids: V3 prefix-sum format. Row b starts with cumulative block count
    # (b+1)*block_num, followed by block ids 0..block_num-1 (read-only K/V blocks).
    torch_bids = torch.zeros((batch, mb), dtype=torch.int64, device=dev_torch)
    for b in range(batch):
        torch_bids[b, 0] = (b + 1) * block_num
        torch_bids[b, 1:mb] = torch.arange(block_num, device=dev_torch)
    block_ids = llaisys.Tensor((batch, mb), dtype=llaisys.DataType.I64,
                               device=llaisys.DeviceType.NVIDIA, device_id=0)
    block_ids.load(c_void_p(torch_bids.data_ptr()))

    # q (decode: seqlen=1 per sample) + shared K/V cache
    q_torch, q = _random_pair((batch, NH, HD), dtype_name)
    k_torch, k_cache = _random_pair((block_num, TOKEN_NUM, NKVH, HD), dtype_name)
    v_torch, v_cache = _random_pair((block_num, TOKEN_NUM, NKVH, HD), dtype_name)

    # cut_idx (batch+1), tot_len (batch)
    cut_torch = torch.arange(batch + 1, dtype=torch.int64, device=dev_torch)
    cut_idx = llaisys.Tensor((batch + 1,), dtype=llaisys.DataType.I64,
                             device=llaisys.DeviceType.NVIDIA, device_id=0)
    cut_idx.load(c_void_p(cut_torch.data_ptr()))

    tot_torch = torch.full((batch,), totlen, dtype=torch.int64, device=dev_torch)
    tot_len = llaisys.Tensor((batch,), dtype=llaisys.DataType.I64,
                             device=llaisys.DeviceType.NVIDIA, device_id=0)
    tot_len.load(c_void_p(tot_torch.data_ptr()))

    # flash_decoding context buffers (total blocks across all seqs)
    total_blocks = batch * block_num
    attn_acc = llaisys.Tensor((total_blocks, NH, HD), dtype=llaisys.DataType.F32,
                              device=llaisys.DeviceType.NVIDIA, device_id=0)
    attn_sum = llaisys.Tensor((total_blocks, NH, 1), dtype=llaisys.DataType.F32,
                              device=llaisys.DeviceType.NVIDIA, device_id=0)
    attn_max = llaisys.Tensor((total_blocks, NH, 1), dtype=llaisys.DataType.F32,
                              device=llaisys.DeviceType.NVIDIA, device_id=0)

    _, attn_val = _random_pair((batch, NH, HD), dtype_name)
    scale = 1.0 / (HD ** 0.5)

    return {
        "q_torch": q_torch, "k_torch": k_torch, "v_torch": v_torch,
        "q": q, "k_cache": k_cache, "v_cache": v_cache,
        "block_ids": block_ids, "cut_idx": cut_idx, "tot_len": tot_len,
        "attn_acc": attn_acc, "attn_sum": attn_sum, "attn_max": attn_max,
        "attn_val": attn_val, "scale": scale,
        "torch_bids": torch_bids, "batch": batch, "totlen": totlen,
    }


def _random_pair(shape, dtype_name):
    t, lt = None, None
    if dtype_name == "f32":
        import torch as _t
        t = _t.rand(shape, dtype=_t.float32, device="cuda:0")
        lt = llaisys.Tensor(shape, dtype=llaisys.DataType.F32,
                            device=llaisys.DeviceType.NVIDIA, device_id=0)
    elif dtype_name == "f16":
        import torch as _t
        t = _t.rand(shape, dtype=_t.float16, device="cuda:0")
        lt = llaisys.Tensor(shape, dtype=llaisys.DataType.F16,
                            device=llaisys.DeviceType.NVIDIA, device_id=0)
    elif dtype_name == "bf16":
        import torch as _t
        t = _t.rand(shape, dtype=_t.bfloat16, device="cuda:0")
        lt = llaisys.Tensor(shape, dtype=llaisys.DataType.BF16,
                            device=llaisys.DeviceType.NVIDIA, device_id=0)
    lt.load(c_void_p(t.data_ptr()))
    return t, lt


def llaisys_flash_decoding(buf: dict) -> None:
    """One batched flash_decoding (decode) call via the paged_attention op."""
    tot_block_num = buf["batch"] * (buf["totlen"] // TOKEN_NUM)
    llaisys.Ops.paged_attention(
        buf["attn_val"], buf["q"], buf["k_cache"], buf["v_cache"], buf["block_ids"],
        buf["cut_idx"], buf["tot_len"], 1, tot_block_num, buf["scale"], False,
        buf["attn_acc"], buf["attn_sum"], buf["attn_max"])


def check_correctness(buf: dict) -> bool:
    """Compare LLAISYS flash_decoding against the torch reference (smallest case)."""
    bids = buf["torch_bids"]
    batch = buf["batch"]
    ref = torch.empty_like(buf["q_torch"])
    prev = 0
    for b in range(batch):
        bnum = int(bids[b, 0]) - prev  # uniform block_num (prefix-sum diff)
        prev = int(bids[b, 0])
        block_ids_values = [int(bids[b, 1 + i]) for i in range(bnum)]
        torch_flash_decoding(ref[b:b + 1],
                             buf["q_torch"][b:b + 1],
                             buf["k_torch"], buf["v_torch"],
                             block_ids_values, buf["scale"], buf["totlen"])
    atol = {"f32": 1e-4, "f16": 2e-2, "bf16": 2e-2}[dtype_name]
    rtol = {"f32": 1e-4, "f16": 2e-2, "bf16": 2e-2}[dtype_name]
    llaisys_flash_decoding(buf)
    torch.cuda.synchronize()
    return check_equal(buf["attn_val"], ref, atol=atol, rtol=rtol)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} + TFLOP")
    parser.add_argument("--dtype", default="bf16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20

    global dtype_name
    dtype_name = args.dtype
    esize = elem_size(dtype_name)

    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Config: nh={NH} nkvh={NKVH} hd={HD} token_num={TOKEN_NUM} dtype={dtype_name}")
    print(f"  FLOPs/iter = batch*{NH}*4*totlen*{HD}")

    results = []
    first = True
    for batch in BATCH_SIZES:
        for totlen in TOTLENS:
            block_num = totlen // TOKEN_NUM
            S = totlen
            total_flops = batch * NH * 4 * S * HD
            # K/V read by every (head, block) instance; Q once; attn_val written.
            bytes_read = (batch * NH * HD + batch * S * NH * 2 * HD) * esize
            bytes_write = batch * NH * HD * esize + batch * block_num * NH * (HD + 2) * 4

            config = BenchmarkConfig(
                op_name=OP_NAME,
                shape={"batch": batch, "totlen": totlen},
                dtype_name=dtype_name,
                warmup=args.warmup, repeat=args.repeat,
                total_bytes_read=bytes_read,
                total_bytes_write=bytes_write,
                total_flops=total_flops,
            )

            # Pre-allocate once; setup_fn returns the same buffers (rotation is
            # unnecessary here — the multi-MB K/V streaming thrashes L2 anyway).
            buf_pool = [make_buffers(batch, totlen, dtype_name) for _ in range(2)]
            _idx = [0]

            def setup_fn():
                b = buf_pool[_idx[0] % 2]
                _idx[0] += 1
                return b

            def llsys_fn(buf): llaisys_flash_decoding(buf)

            if first:
                ok = check_correctness(buf_pool[0])
                print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
                assert ok, "flash_decoding correctness check failed!"
                first = False

            result = run_benchmark(config, setup_fn, llsys_fn)
            results.append(result)

    save_results(results, args.output, SHAPE_LABEL)


if __name__ == "__main__":
    main()
