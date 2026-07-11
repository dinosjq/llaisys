#!/usr/bin/env python3
"""Benchmark: kv-cache-move operator (writes flat K/V into paged blocks, no baseline)."""

import sys, os, math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from ctypes import c_void_p
import torch
import llaisys
from test_utils import random_tensor, zero_tensor, torch_device, llaisys_device, llaisys_dtype

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "kv_cache_move"
KV_CACHE_TOKEN_NUM = 64
QWEN2_SHAPES = [
    {"batch": 1, "max_seq_len": 64,  "nkvh": 4, "dh": 128},
    {"batch": 1, "max_seq_len": 512, "nkvh": 4, "dh": 128},
    {"batch": 4, "max_seq_len": 64,  "nkvh": 4, "dh": 128},
    {"batch": 8, "max_seq_len": 128, "nkvh": 4, "dh": 128},
]


# ---------------------------------------------------------------------------
# helpers for constructing I64 tensors
# ---------------------------------------------------------------------------

def make_i64_tensor(data, shape, device_name="nvidia", device_id=0) -> tuple:
    """Create a torch+llaisys I64 tensor pair from a list/1-D array."""
    t = torch.tensor(data, dtype=torch.int64,
                     device=torch_device(device_name, device_id))
    ll = llaisys.Tensor(
        shape, dtype=llaisys.DataType.I64,
        device=llaisys_device(device_name), device_id=device_id)
    ll.load(c_void_p(t.data_ptr()))
    return t, ll


# ---------------------------------------------------------------------------
# benchmark setup
# ---------------------------------------------------------------------------

def setup_kv_cache_move(batch: int, max_seq_len: int, nkvh: int, dh: int,
                        dtype_name: str) -> dict:
    """Allocate flat input, blocked output, block-id table, and position meta.

    out (blocked) : (batch * max_block_num, token_num, nkvh, dh)
    in  (flat)    : (batch * max_seq_len, nkvh, dh)
    block_ids     : (batch, max_block_num)  dtype I64
    cut_idx       : (batch + 1,)            dtype I64
    pos_ids       : (batch * max_seq_len,)  dtype I64
    """
    token_num = KV_CACHE_TOKEN_NUM
    max_block_num = math.ceil(max_seq_len / token_num) + 1
    total_seq_len = batch * max_seq_len
    numel = nkvh * dh
    device_id = 0

    # out — blocked KV cache (batch * max_block_num blocks)
    out_torch, out = random_tensor(
        (batch * max_block_num, token_num, nkvh, dh),
        dtype_name, "nvidia", device_id=device_id)

    # in — flat K/V tensor
    in_torch, in_ll = random_tensor(
        (total_seq_len, nkvh, dh),
        dtype_name, "nvidia", device_id=device_id)

    # block_ids  (batch, max_block_num) — each batch gets its own block range
    block_ids_data = []
    for b in range(batch):
        base = b * max_block_num
        block_ids_data.append([base + j for j in range(max_block_num)])
    block_ids_t, block_ids_ll = make_i64_tensor(
        block_ids_data, (batch, max_block_num))

    # cut_idx  (batch + 1,) — even split
    cut_idx_data = [b * max_seq_len for b in range(batch + 1)]
    cut_idx_t, cut_idx_ll = make_i64_tensor(cut_idx_data, (batch + 1,))

    # pos_ids  (total_seq_len,) — monotonic within each batch segment
    pos_ids_data = []
    for b in range(batch):
        pos_ids_data.extend(range(max_seq_len))
    pos_ids_t, pos_ids_ll = make_i64_tensor(
        pos_ids_data, (total_seq_len,))

    return {
        "out_torch": out_torch, "out": out,
        "in_torch": in_torch, "in": in_ll,
        "block_ids_torch": block_ids_t, "block_ids": block_ids_ll,
        "cut_idx_torch": cut_idx_t, "cut_idx": cut_idx_ll,
        "pos_ids_torch": pos_ids_t, "pos_ids": pos_ids_ll,
        "max_seq_len": max_seq_len,
        "max_block_num": max_block_num,
        "token_num": token_num,
        "total_seq_len": total_seq_len,
        "batch": batch,
        "numel": numel,
    }


def llaisys_kv_cache_move(buf: dict) -> None:
    llaisys.Ops.kv_cache_move(
        buf["out"], buf["in"], buf["block_ids"],
        buf["cut_idx"], buf["pos_ids"], buf["max_seq_len"],
    )


# ---------------------------------------------------------------------------
# correctness check
# ---------------------------------------------------------------------------

def check_correctness(buf: dict, dtype_name: str) -> bool:
    """Verify that kv_cache_move copied each flat position to the correct
    (block_id, token_id) slot in the blocked output.

    Runs the op, then fetches both in and out back to torch and checks
    element-by-element.
    """
    # Snapshot input before the move
    in_torch = buf["in_torch"].clone()

    # Snapshot output before (to detect stale overwrites)
    # ... not needed; we check only that the correct data arrived.

    llaisys_kv_cache_move(buf)
    torch.cuda.synchronize()

    # Fetch llaisys out tensor into torch
    out_view = _fetch_llaisys(buf["out"])
    in_before = in_torch

    batch = buf["batch"]
    max_seq_len = buf["max_seq_len"]
    token_num = buf["token_num"]
    max_block_num = buf["max_block_num"]
    numel = buf["numel"]

    block_ids_data = []
    for b in range(batch):
        base = b * max_block_num
        block_ids_data.append([base + j for j in range(max_block_num)])

    cut_idx_data = [b * max_seq_len for b in range(batch + 1)]

    pos_ids_data = []
    for b in range(batch):
        pos_ids_data.extend(range(max_seq_len))

    # Check each position
    for b in range(batch):
        lo = cut_idx_data[b]
        up = cut_idx_data[b + 1]
        for offset in range(up - lo):
            seq_id = lo + offset
            pos_id = pos_ids_data[seq_id]
            block_id = block_ids_data[b][pos_id // token_num]
            token_id = pos_id % token_num

            expected = in_before[seq_id]  # (nkvh, dh)
            actual = out_view[block_id, token_id]  # (nkvh, dh)

            # Compare
            atol = 1e-3 if dtype_name in ("f16", "bf16") else 1e-5
            rtol = 1e-3 if dtype_name in ("f16", "bf16") else 1e-5
            if not torch.allclose(expected, actual, atol=atol, rtol=rtol):
                print(f"  Mismatch at batch={b} offset={offset} "
                      f"block_id={block_id} token_id={token_id}")
                return False
    return True


_API_CACHE = None


def _fetch_llaisys(t: llaisys.Tensor) -> torch.Tensor:
    """Copy a contiguous LLAISYS tensor back to a torch tensor."""
    global _API_CACHE
    shape = t.shape()
    dtype = t.dtype()
    device_type = t.device_type()
    device_id = t.device_id()

    dtype_map = {
        llaisys.DataType.F16: torch.float16,
        llaisys.DataType.F32: torch.float32,
        llaisys.DataType.BF16: torch.bfloat16,
        llaisys.DataType.I64: torch.int64,
        llaisys.DataType.I32: torch.int32,
    }
    out = torch.empty(shape, dtype=dtype_map.get(dtype, torch.float32),
                      device=torch_device("nvidia", device_id))

    if _API_CACHE is None:
        _API_CACHE = llaisys.RuntimeAPI(device_type)
    _API_CACHE.memcpy_sync(
        out.data_ptr(), t.data_ptr(),
        out.numel() * out.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return out


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
        batch = shape_info["batch"]
        max_seq_len = shape_info["max_seq_len"]
        nkvh = shape_info["nkvh"]
        dh = shape_info["dh"]
        esize = elem_size(dtype_name)
        token_num = KV_CACHE_TOKEN_NUM
        max_block_num = math.ceil(max_seq_len / token_num) + 1
        total_seq_len = batch * max_seq_len
        numel = nkvh * dh

        # in data
        total_bytes_read = total_seq_len * numel * esize
        # block_ids + cut_idx + pos_ids (all I64 = 8 bytes each)
        meta_bytes = (batch * max_block_num + (batch + 1) + total_seq_len) * 8
        total_bytes_read += meta_bytes
        # out data
        total_bytes_write = batch * max_block_num * token_num * numel * esize
        # Pure memory copy -- no FLOPs
        total_flops = 0

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape=shape_info,
            dtype_name=dtype_name,
            warmup=args.warmup, repeat=args.repeat,
            total_bytes_read=total_bytes_read,
            total_bytes_write=total_bytes_write,
            total_flops=total_flops,
        )

        def setup_fn(b=batch, msl=max_seq_len, n=nkvh, d=dh, dt=dtype_name):
            return setup_kv_cache_move(b, msl, n, d, dt)
        def llsys_fn(buf): llaisys_kv_cache_move(buf)

        # No baseline: ref_fn=None
        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn=None)
        # baseline_name stays empty
        results.append(result)

        if shape_info == QWEN2_SHAPES[0]:
            buf = setup_kv_cache_move(batch, max_seq_len, nkvh, dh, dtype_name)
            ok = check_correctness(buf, dtype_name)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "KV-cache-move correctness check failed!"


    save_results(results, args.output)


if __name__ == "__main__":
    main()
