import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from test_utils import (
    random_tensor,
    zero_tensor,
    benchmark,
    torch_device,
    device_name,
    torch_dtype,
    dtype_name as llaisys_dtype_name,
)


def fetch_llaisys_tensor(t: llaisys.Tensor) -> torch.Tensor:
    shape = t.shape()
    strides = t.strides()

    right = 0
    for i in range(len(shape)):
        if strides[i] > 0:
            right += strides[i] * (shape[i] - 1)
        else:
            raise ValueError("Negative strides are not supported yet")

    out = torch.empty(
        (right + 1,),
        dtype=torch_dtype(llaisys_dtype_name(t.dtype())),
        device=torch_device(device_name(t.device_type()), t.device_id()),
    )
    out_view = torch.as_strided(out, shape, strides)

    api = llaisys.RuntimeAPI(t.device_type())
    bytes_ = (right + 1) * out.element_size()
    api.memcpy_sync(out.data_ptr(), t.data_ptr(), bytes_, llaisys.MemcpyKind.D2D)
    return out_view


def torch_topk(vals: torch.Tensor, k: int):
    return torch.topk(vals, k, dim=-1, largest=True, sorted=True)


def test_op_topk(
    n: int,
    k: int,
    dtype_name: str = "f32",
    device_name_: str = "cpu",
    profile: bool = False,
):
    print(f"   n={n} k={k} dtype <{dtype_name}>")
    assert 0 < k <= n

    if dtype_name == "f32":
        atol, rtol = 1e-5, 1e-5
    elif dtype_name == "f16":
        atol, rtol = 1e-3, 1e-3
    elif dtype_name == "bf16":
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    vals, vals_ = random_tensor((n,), dtype_name, device_name_)
    out_idx, out_idx_ = zero_tensor((k,), "i64", device_name_)
    out_val, out_val_ = zero_tensor((k,), dtype_name, device_name_)

    ref_val, ref_idx = torch_topk(vals, k)

    llaisys.Ops.topk(out_idx_, out_val_, vals_, k)

    got_idx = fetch_llaisys_tensor(out_idx_).to(torch.int64)
    got_val = fetch_llaisys_tensor(out_val_).to(vals.dtype)

    # Basic sanity checks
    assert got_idx.shape == (k,)
    assert got_val.shape == (k,)
    assert int(got_idx.min()) >= 0
    assert int(got_idx.max()) < n
    assert got_idx.unique().numel() == k

    # Indices and values should match the input
    gathered = vals[got_idx]
    assert torch.allclose(gathered, got_val, atol=atol, rtol=rtol)

    # Order may differ when ties exist; compare value multiset
    got_sorted = torch.sort(got_val, descending=True).values
    assert torch.allclose(got_sorted, ref_val, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_topk(vals, k),
            lambda: llaisys.Ops.topk(out_idx_, out_val_, vals_, k),
            device_name_,
        )


def test_op_topk_batched(
    batch: int,
    n: int,
    k: int,
    dtype_name: str = "f32",
    device_name_: str = "cpu",
    profile: bool = False,
):
    print(f"   batch={batch} n={n} k={k} dtype <{dtype_name}>")
    assert 0 < k <= n

    if dtype_name == "f32":
        atol, rtol = 1e-5, 1e-5
    elif dtype_name == "f16":
        atol, rtol = 1e-3, 1e-3
    elif dtype_name == "bf16":
        atol, rtol = 1e-2, 1e-2
    else:
        raise ValueError(f"Unsupported dtype: {dtype_name}")

    vals, vals_ = random_tensor((batch, n), dtype_name, device_name_)
    out_idx, out_idx_ = zero_tensor((batch, k), "i64", device_name_)
    out_val, out_val_ = zero_tensor((batch, k), dtype_name, device_name_)

    ref_val, ref_idx = torch_topk(vals, k)

    llaisys.Ops.topk(out_idx_, out_val_, vals_, k)

    got_idx = fetch_llaisys_tensor(out_idx_).to(torch.int64)
    got_val = fetch_llaisys_tensor(out_val_).to(vals.dtype)

    # Basic sanity checks
    assert got_idx.shape == (batch, k)
    assert got_val.shape == (batch, k)
    assert int(got_idx.min()) >= 0
    assert int(got_idx.max()) < n

    # Per-row checks: indices/value consistency and value multiset vs torch
    for b in range(batch):
        gathered = vals[b, got_idx[b]]
        assert torch.allclose(gathered, got_val[b], atol=atol, rtol=rtol)

        got_sorted = torch.sort(got_val[b], descending=True).values
        assert torch.allclose(got_sorted, ref_val[b], atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_topk(vals, k),
            lambda: llaisys.Ops.topk(out_idx_, out_val_, vals_, k),
            device_name_,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_cases = [
        # (n, k)
        (16, 1),
        (16, 5),
        (4096, 32),
    ]
    test_dtypes = ["f32", "f16", "bf16"]

    print(f"Testing Ops.topk on {args.device}")
    for n, k in test_cases:
        for dtype_name in test_dtypes:
            test_op_topk(n, k, dtype_name, args.device, args.profile)
    # batched cases
    batched_cases = [
        (4, 16, 1),
        (4, 16, 5),
        (2, 4096, 32),
    ]
    for batch, n, k in batched_cases:
        for dtype_name in test_dtypes:
            test_op_topk_batched(batch, n, k, dtype_name, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
