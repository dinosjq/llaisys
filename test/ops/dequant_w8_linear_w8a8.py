"""NVIDIA: linear_w8a8 (W8A8 cuBLAS INT8 GEMM + Hadamard) vs 原始权重 FP linear."""

# quantize_w8 / quantize_act 在量化前做 K 维 Hadamard 旋转（x·H 与 H·W，乘积 x·W 不变），
# 因此参考直接用原始权重 FP linear 即可（误差即 Hadamard 后的 INT8 量化误差）。

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


def test_linear_w8a8_vs_fp_linear(dtype_name="bf16", m=4, n=8, k=16):
    device_name = "nvidia"
    _, w_fp_ = random_tensor((n, k), dtype_name, device_name, scale=0.05)

    w_i8 = llaisys.Tensor([n, k], dtype=llaisys.DataType.I8, device=llaisys_device(device_name), device_id=0)
    scale_t = llaisys.Tensor([n], dtype=llaisys.DataType.F32, device=llaisys_device(device_name), device_id=0)
    llaisys.Ops.quantize_w8(w_i8, scale_t, w_fp_)

    _, x_ = random_tensor((m, k), dtype_name, device_name, scale=0.1)
    _, out_ref_ = random_tensor((m, n), dtype_name, device_name)
    _, out_w8_ = random_tensor((m, n), dtype_name, device_name)
    # 参考：原始权重 FP linear（Hadamard 旋转不改变数学结果）
    llaisys.Ops.linear(out_ref_, x_, w_fp_, None)

    # 独立激活量化步骤（含 Hadamard 旋转）+ W8A8 量化线性
    in_q = llaisys.Tensor([m, k], dtype=llaisys.DataType.I8, device=llaisys_device(device_name), device_id=0)
    a_scale = llaisys.Tensor([m], dtype=llaisys.DataType.F32, device=llaisys_device(device_name), device_id=0)
    llaisys.Ops.quantize_act(in_q, a_scale, x_)
    llaisys.Ops.linear_w8a8(out_w8_, in_q, a_scale, w_i8, scale_t, None)

    ref = _llaisys_to_torch(out_ref_)
    # W8A8 同时量化激活与权重（Hadamard 旋转后误差应显著小于未旋转）
    atol = 2e-1 if dtype_name != "f32" else 1e-1
    rtol = 2e-1 if dtype_name != "f32" else 1e-1
    assert check_equal(out_w8_, ref, atol=atol, rtol=rtol)
    print(f"  ok dtype={dtype_name} m={m} n={n} k={k}")


if __name__ == "__main__":
    print("Testing W8A8 linear_w8a8 (cuBLAS INT8) on nvidia")
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
        test_linear_w8a8_vs_fp_linear(dtype, 4, 8, 16)
        for m, n, k in cases:
            test_linear_w8a8_vs_fp_linear(dtype, m, n, k)
    print("Passed.")
