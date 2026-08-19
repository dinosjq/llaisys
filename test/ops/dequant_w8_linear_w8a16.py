"""NVIDIA: linear_w8a16 (gemv + large-M) vs dequant_w8 + linear."""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys
from test_utils import (
    random_tensor,
    check_equal,
    llaisys_device,
    torch_dtype,
    torch_device,
    device_name as device_name_from_enum,
    dtype_name as dtype_name_from_enum,
)


def _llaisys_to_torch(t: llaisys.Tensor) -> torch.Tensor:
    shape = t.shape()
    out = torch.empty(
        shape,
        dtype=torch_dtype(dtype_name_from_enum(t.dtype())),
        device=torch_device(device_name_from_enum(t.device_type()), t.device_id()),
    )
    api = llaisys.RuntimeAPI(t.device_type())
    api.memcpy_sync(
        out.data_ptr(),
        t.data_ptr(),
        out.numel() * out.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return out


def test_linear_w8a16_vs_dequant_linear(dtype_name="bf16", m=4, n=8, k=16):
    device_name = "nvidia"
    _, w_fp_ = random_tensor((n, k), dtype_name, device_name, scale=0.05)

    w_i8 = llaisys.Tensor([n, k], dtype=llaisys.DataType.I8, device=llaisys_device(device_name), device_id=0)
    scale_t = llaisys.Tensor([n], dtype=llaisys.DataType.F32, device=llaisys_device(device_name), device_id=0)
    llaisys.Ops.quantize_w8(w_i8, scale_t, w_fp_)

    _, deq_ = random_tensor((n, k), dtype_name, device_name)
    llaisys.Ops.dequant_w8(deq_, w_i8, scale_t)

    _, x_ = random_tensor((m, k), dtype_name, device_name, scale=0.1)
    _, out_ref_ = random_tensor((m, n), dtype_name, device_name)
    _, out_w8_ = random_tensor((m, n), dtype_name, device_name)
    llaisys.Ops.linear(out_ref_, x_, deq_, None)
    llaisys.Ops.linear_w8a16(out_w8_, x_, w_i8, scale_t, None)

    ref = _llaisys_to_torch(out_ref_)
    # One-stage dp4a path also quantizes activations; allow extra slack vs pure W8A16.
    atol = 1e-1 if dtype_name != "f32" else 5e-2
    rtol = 1e-1 if dtype_name != "f32" else 5e-2
    assert check_equal(out_w8_, ref, atol=atol, rtol=rtol)
    print(f"  ok dtype={dtype_name} m={m} n={n} k={k}")


if __name__ == "__main__":
    print("Testing W8 hybrid linear_w8a16 on nvidia")
    cases = [
        (1, 1536, 1536),
        (4, 1536, 1536),
        (8, 8960, 1536),
        (15, 1536, 1536),
        (16, 1536, 1536),
        (128, 1536, 1536),
        (1, 3072, 3072),
        (8, 8192, 3072),
    ]
    for dtype in ("bf16", "f32"):
        test_linear_w8a16_vs_dequant_linear(dtype, 4, 8, 16)
        for m, n, k in cases:
            test_linear_w8a16_vs_dequant_linear(dtype, m, n, k)
    print("Passed.")
