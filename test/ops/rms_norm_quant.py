"""NVIDIA: rms_norm_quant（融合 rms_norm+Hadamard+量化） vs rms_norm + quantize_act 两步参考。"""

import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import torch
import llaisys
from test_utils import random_tensor, torch_dtype, torch_device, device_name, dtype_name, llaisys_device


def to_torch(t: llaisys.Tensor) -> torch.Tensor:
    shape = t.shape()
    if llaisys.DataType(t.dtype()) == llaisys.DataType.I8:
        dt = torch.int8
    else:
        dt = torch_dtype(dtype_name(t.dtype()))
    out = torch.empty(shape, dtype=dt, device=torch_device(device_name(t.device_type()), t.device_id()))
    api = llaisys.RuntimeAPI(t.device_type())
    api.memcpy_sync(out.data_ptr(), t.data_ptr(), out.numel() * out.element_size(), llaisys.MemcpyKind.D2D)
    return out


def test_rms_norm_quant(dtype="bf16", m=8, k=1536, eps=1e-5):
    device = "nvidia"
    x, x_ = random_tensor((m, k), dtype, device, scale=0.5)
    w, w_ = random_tensor((k,), dtype, device, scale=0.5)
    dev = llaisys_device(device)

    # 参考两步：rms_norm → quantize_act
    x_norm = llaisys.Tensor([m, k], dtype=x_.dtype(), device=dev, device_id=0)
    llaisys.Ops.rms_norm(x_norm, x_, w_, eps)
    q_ref = llaisys.Tensor([m, k], dtype=llaisys.DataType.I8, device=dev, device_id=0)
    s_ref = llaisys.Tensor([m], dtype=llaisys.DataType.F32, device=dev, device_id=0)
    llaisys.Ops.quantize_act(q_ref, s_ref, x_norm)

    # 融合：rms_norm_quant
    q = llaisys.Tensor([m, k], dtype=llaisys.DataType.I8, device=dev, device_id=0)
    s = llaisys.Tensor([m], dtype=llaisys.DataType.F32, device=dev, device_id=0)
    llaisys.Ops.rms_norm_quant(q, s, x_, w_, eps)

    q_ref_t, q_t = to_torch(q_ref), to_torch(q)
    s_ref_t, s_t = to_torch(s_ref), to_torch(s)
    # int8 允许边界 1 LSB 差异（内部 rms_norm 归约顺序不同 → 极个别舍入翻边）
    assert torch.allclose(q_t.float(), q_ref_t.float(), atol=1, rtol=0), "rms_norm_quant int8 mismatch"
    assert torch.allclose(s_t, s_ref_t, atol=1e-4, rtol=1e-4), "rms_norm_quant scale mismatch"
    print(f"  ok dtype={dtype} m={m} k={k}")


if __name__ == "__main__":
    print("Testing rms_norm_quant (fused rms_norm+Hadamard+quant) on nvidia")
    for dtype in ("bf16", "f16", "f32"):
        test_rms_norm_quant(dtype, 8, 1536)
        test_rms_norm_quant(dtype, 32, 1536)
        test_rms_norm_quant(dtype, 8, 3072)
    print("\033[92mTest passed!\033[0m\n")
