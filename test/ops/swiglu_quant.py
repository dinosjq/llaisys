"""NVIDIA: swiglu_quant（融合 swiglu+Hadamard+量化） vs swiglu + quantize_act 两步参考。"""

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


def test_swiglu_quant(dtype="bf16", m=8, k=8960):
    device = "nvidia"
    gate, gate_ = random_tensor((m, k), dtype, device, scale=0.3)
    up, up_ = random_tensor((m, k), dtype, device, scale=0.3)
    dev = llaisys_device(device)

    # 参考两步：swiglu → quantize_act
    sw = llaisys.Tensor([m, k], dtype=gate_.dtype(), device=dev, device_id=0)
    llaisys.Ops.swiglu(sw, gate_, up_)
    q_ref = llaisys.Tensor([m, k], dtype=llaisys.DataType.I8, device=dev, device_id=0)
    s_ref = llaisys.Tensor([m], dtype=llaisys.DataType.F32, device=dev, device_id=0)
    llaisys.Ops.quantize_act(q_ref, s_ref, sw)

    # 融合：swiglu_quant
    q = llaisys.Tensor([m, k], dtype=llaisys.DataType.I8, device=dev, device_id=0)
    s = llaisys.Tensor([m], dtype=llaisys.DataType.F32, device=dev, device_id=0)
    llaisys.Ops.swiglu_quant(q, s, gate_, up_)

    q_ref_t, q_t = to_torch(q_ref), to_torch(q)
    s_ref_t, s_t = to_torch(s_ref), to_torch(s)
    # int8 允许边界 1 LSB 差异
    assert torch.allclose(q_t.float(), q_ref_t.float(), atol=1, rtol=0), "swiglu_quant int8 mismatch"
    assert torch.allclose(s_t, s_ref_t, atol=1e-4, rtol=1e-4), "swiglu_quant scale mismatch"
    print(f"  ok dtype={dtype} m={m} k={k}")


if __name__ == "__main__":
    print("Testing swiglu_quant (fused swiglu+Hadamard+quant) on nvidia")
    for dtype in ("bf16", "f16", "f32"):
        test_swiglu_quant(dtype, 8, 8960)
        test_swiglu_quant(dtype, 32, 8960)
        test_swiglu_quant(dtype, 8, 8192)
    print("\033[92mTest passed!\033[0m\n")
