#!/usr/bin/env python3
"""Benchmark: paged-attention operator (block-based KV-cache attention, no baseline)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ctypes import c_void_p
import torch
import llaisys
from test_utils import random_tensor, check_equal, torch_device, llaisys_device, llaisys_dtype

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "paged_attention"
KV_CACHE_TOKEN_NUM = 64
QWEN2_SHAPES = [
    {"seqlen": 16,  "token_num": 64, "block_num": 8,
     "block_ids": [0, 1, 2, 3], "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 128, "token_num": 64, "block_num": 8,
     "block_ids": [0, 1, 2, 3, 4, 5, 6, 7], "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 512, "token_num": 64, "block_num": 16,
     "block_ids": list(range(16)), "nh": 28, "nkvh": 4, "hd": 128},
]


# ---------------------------------------------------------------------------
# reference implementation (copied & adapted from test/ops/paged_attention.py)
# ---------------------------------------------------------------------------

def torch_paged_attention(attn_val, query, k_cache, v_cache, block_ids, scale, totlen=None):
    """Dense-attention reference that expands the paged cache into a flat K/V."""
    seqlen, nh, hd = query.shape
    _, token_num, nkvh, _ = k_cache.shape
    if totlen is None:
        totlen = len(block_ids) * token_num
    prefix_len = totlen - seqlen

    dense_k = torch.empty((totlen, nkvh, hd), dtype=query.dtype, device=query.device)
    dense_v = torch.empty((totlen, nkvh, hd), dtype=query.dtype, device=query.device)
    for token_id in range(totlen):
        block_id = int(block_ids[token_id // token_num])
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
    key_ids = torch.arange(totlen, device=query.device)
    query_ids = torch.arange(seqlen, device=query.device).unsqueeze(1)
    mask = key_ids.unsqueeze(0) <= (prefix_len + query_ids)
    attn_weight = attn_weight.masked_fill(~mask.unsqueeze(0), float("-inf"))
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_val.copy_(torch.matmul(attn_weight, v).transpose(0, 1))


def build_block_ids(values, device_name="nvidia", device_id=0):
    """Build a 1-D I32 block_ids tensor for a single sample."""
    torch_device_obj = torch.device(f"cuda:{device_id}") if device_name == "nvidia" else torch.device("cpu")
    torch_ids = torch.tensor(values, dtype=torch.int32, device=torch_device_obj)
    block_ids_ll = llaisys.Tensor(
        (len(values),),
        dtype=llaisys.DataType.I32,
        device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
        device_id=device_id,
    )
    block_ids_ll.load(c_void_p(torch_ids.data_ptr()))
    return torch_ids, block_ids_ll


# ---------------------------------------------------------------------------
# benchmark setup
# ---------------------------------------------------------------------------

def setup_paged_attention(seqlen: int, token_num: int, block_num: int,
                          block_ids_values: list, nh: int, nkvh: int, hd: int,
                          dtype_name: str) -> dict:
    """Allocate Q, K/V cache blocks, block-id table, and output.

    Uses the batched single-shot API with batch=1.
    """
    device_id = 0
    totlen = len(block_ids_values) * token_num
    batch = 1
    max_block_num = block_num  # the padded block-id table width

    # q  : (seqlen, nh, hd)
    q_torch, q = random_tensor((seqlen, nh, hd), dtype_name, "nvidia", device_id=device_id)
    # k_cache : (block_num, token_num, nkvh, hd)
    k_cache_torch, k_cache = random_tensor(
        (block_num, token_num, nkvh, hd), dtype_name, "nvidia", device_id=device_id)
    # v_cache : (block_num, token_num, nkvh, hd)
    v_cache_torch, v_cache = random_tensor(
        (block_num, token_num, nkvh, hd), dtype_name, "nvidia", device_id=device_id)
    # attn_val : (seqlen, nh, hd)
    attn_val_torch, attn_val = random_tensor(
        (seqlen, nh, hd), dtype_name, "nvidia", device_id=device_id)

    # block_ids 2-D table  (batch=1, max_block_num)  dtype I32
    row = list(block_ids_values) + [0] * (max_block_num - len(block_ids_values))
    block_ids_t = torch.tensor([row], dtype=torch.int32,
                               device=torch_device("nvidia", device_id))
    block_ids_ll = llaisys.Tensor(
        (batch, max_block_num), dtype=llaisys.DataType.I32,
        device=llaisys_device("nvidia"), device_id=device_id)
    block_ids_ll.load(c_void_p(block_ids_t.data_ptr()))

    # cut_idx  (batch+1,)  dtype I32
    cut_idx_t = torch.tensor([0, seqlen], dtype=torch.int32,
                             device=torch_device("nvidia", device_id))
    cut_idx_ll = llaisys.Tensor(
        (batch + 1,), dtype=llaisys.DataType.I32,
        device=llaisys_device("nvidia"), device_id=device_id)
    cut_idx_ll.load(c_void_p(cut_idx_t.data_ptr()))

    # tot_len per sample  (batch,)  dtype I32
    totlen_t = torch.tensor([totlen], dtype=torch.int32,
                            device=torch_device("nvidia", device_id))
    totlen_ll = llaisys.Tensor(
        (batch,), dtype=llaisys.DataType.I32,
        device=llaisys_device("nvidia"), device_id=device_id)
    totlen_ll.load(c_void_p(totlen_t.data_ptr()))

    max_seq_len = seqlen  # max across batch (only one sample)
    scale = 1.0 / (hd ** 0.5)

    return {
        "q_torch": q_torch, "q": q,
        "k_cache_torch": k_cache_torch, "k_cache": k_cache,
        "v_cache_torch": v_cache_torch, "v_cache": v_cache,
        "attn_val_torch": attn_val_torch, "attn_val": attn_val,
        "block_ids_torch": block_ids_t, "block_ids": block_ids_ll,
        "block_ids_values": block_ids_values,
        "cut_idx_torch": cut_idx_t, "cut_idx": cut_idx_ll,
        "totlen_torch": totlen_t, "tot_len": totlen_ll,
        "max_seq_len": max_seq_len,
        "scale": scale,
        "seqlen": seqlen, "totlen": totlen,
        "nh": nh, "nkvh": nkvh, "hd": hd,
    }


def llaisys_paged_attention(buf: dict) -> None:
    llaisys.Ops.paged_attention(
        buf["attn_val"], buf["q"], buf["k_cache"], buf["v_cache"],
        buf["block_ids"], buf["cut_idx"], buf["tot_len"],
        buf["max_seq_len"], buf["scale"],
    )


# ---------------------------------------------------------------------------
# correctness check (runs once)
# ---------------------------------------------------------------------------

def check_correctness(buf: dict, dtype_name: str) -> bool:
    """Compare LLAISYS output against the torch dense-attention reference."""
    torch_paged_attention(
        buf["attn_val_torch"], buf["q_torch"],
        buf["k_cache_torch"], buf["v_cache_torch"],
        buf["block_ids_values"], buf["scale"],
    )
    torch_ref = buf["attn_val_torch"].clone()

    llaisys_paged_attention(buf)

    # Use wider tolerances for fp16
    if dtype_name == "f32":
        atol, rtol = 1e-3, 1e-3
    elif dtype_name == "bf16":
        atol, rtol = 1e-2, 1e-2
    else:
        atol, rtol = 1e-2, 1e-2
    return check_equal(buf["attn_val"], torch_ref, atol=atol, rtol=rtol)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

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
        token_num = shape_info["token_num"]
        block_num = shape_info["block_num"]
        block_ids_values = shape_info["block_ids"]
        nh = shape_info["nh"]
        nkvh = shape_info["nkvh"]
        hd = shape_info["hd"]
        esize = elem_size(dtype_name)
        totlen = len(block_ids_values) * token_num

        # bytes read: q + k_cache blocks + v_cache blocks  (~ all blocks)
        bytes_q = seqlen * nh * hd * esize
        bytes_kv = block_num * token_num * nkvh * hd * esize * 2
        total_bytes_read = bytes_q + bytes_kv
        # bytes write: attn_val
        total_bytes_write = seqlen * nh * hd * esize
        # FLOPs: same as self-attention
        total_flops = 4 * seqlen * totlen * nh * hd

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape=shape_info,
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(sl=seqlen, tn=token_num, bn=block_num, bids=block_ids_values,
                     n=nh, nk=nkvh, h=hd, d=dtype_name):
            return setup_paged_attention(sl, tn, bn, bids, n, nk, h, d)
        def llsys_fn(buf): llaisys_paged_attention(buf)

        # No baseline: ref_fn=None
        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn=None)
        # baseline_name stays empty
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_paged_attention(
                seqlen, token_num, block_num, block_ids_values, nh, nkvh, hd, dtype_name)
            ok = check_correctness(buf, dtype_name)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Paged-attention correctness check failed!"


    save_results(results, args.output)


if __name__ == "__main__":
    main()
