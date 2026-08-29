"""KV cache INT8 量化链路测试（NVIDIA only）。

覆盖：
  1. kv_cache_move 量化写入：FP K/V → I8 cache + F32 per-(token,nkvh) scale
  2. paged_attention prefill 量化读取：量化 cache 输入 → 输出 ≈ FP 参考
  3. flash_decoding decode 量化读取：量化 cache 输入 → 输出 ≈ FP 参考
"""
import argparse
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import llaisys
import torch
from ctypes import c_void_p

from test_utils import random_tensor, check_equal, torch_device, llaisys_device, llaisys_dtype


# ── 通用 helper ──────────────────────────────────────────────

def torch_quantize_cache(cache_fp: torch.Tensor):
    """per-(token,nkvh) 行 absmax 量化 → (i8, scale) 。全部在 float 精度计算，与 kernel 一致。"""
    f = cache_fp.float()
    amax = f.abs().amax(dim=-1, keepdim=True)
    scale = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
    i8 = torch.clamp(torch.round(f / scale), -128, 127).to(torch.int8)
    return i8, scale


def copy_to_llaisys(value: torch.Tensor, dtype, device):
    tensor = llaisys.Tensor(value.shape, dtype=dtype, device=device)
    api = llaisys.RuntimeAPI(device)
    api.memcpy_sync(
        tensor.data_ptr(),
        value.data_ptr(),
        value.numel() * value.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return tensor


def copy_from_llaisys(value, dtype: torch.dtype) -> torch.Tensor:
    result = torch.empty(value.shape(), dtype=dtype, device=torch_device("nvidia", 0))
    api = llaisys.RuntimeAPI(value.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        value.data_ptr(),
        result.numel() * result.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return result


def build_block_ids(values, device_name="nvidia", device_id=0):
    """V3-format block_ids：[count, id_0, id_1, ...]"""
    tdev = torch_device(device_name, device_id)
    count = len(values)
    padded = [count] + list(values)
    torch_ids = torch.tensor(padded, dtype=torch.int64, device=tdev)
    block_ids = llaisys.Tensor((1, len(padded)), dtype=llaisys.DataType.I64,
                               device=llaisys_device(device_name), device_id=device_id)
    block_ids.load(c_void_p(torch_ids.data_ptr()))
    return torch_ids, block_ids


# ── 1. 量化写入 ─────────────────────────────────────────────

def torch_quant_write_reference(inp, block_ids, cut_idx, pos_ids, token_num, nkvh, dh):
    """按 block 表写出量化后的 (i8, scale)，与 kernel 布局一致。"""
    block_num = int(block_ids[:, 1:].max()) + 1
    i8 = torch.zeros((block_num, token_num, nkvh, dh), dtype=torch.int8, device=inp.device)
    scale = torch.zeros((block_num, token_num, nkvh), dtype=torch.float32, device=inp.device)
    for batch in range(block_ids.shape[0]):
        lo, up = int(cut_idx[batch]), int(cut_idx[batch + 1])
        for seq_id in range(lo, up):
            pos_id = int(pos_ids[seq_id])
            block_id = int(block_ids[batch, pos_id // token_num + 1])
            local = pos_id % token_num
            row = inp[seq_id].float()
            amax = row.abs().amax(dim=-1, keepdim=True)
            s = torch.where(amax == 0, torch.ones_like(amax), amax / 127.0)
            i8[block_id, local] = torch.clamp(torch.round(row / s), -128, 127).to(torch.int8)
            scale[block_id, local] = s[..., 0]
    return i8, scale


def test_quant_write(dtype_name: str):
    print(f"  [quant-write] dtype <{dtype_name}>")
    torch_dtype = {"f32": torch.float32, "f16": torch.float16, "bf16": torch.bfloat16}[dtype_name]
    dev = torch.device("cuda:0")
    token_num, nkvh, dh = 4, 2, 8
    inp = (torch.randn(5, nkvh, dh, device=dev) * 2.0).to(torch_dtype)
    block_ids_t = torch.tensor([[2, 5, 1], [4, 6, 3]], dtype=torch.int64, device=dev)
    cut_idx_t = torch.tensor([0, 2, 5], dtype=torch.int64, device=dev)
    pos_ids_t = torch.tensor([5, 6, 1, 2, 7], dtype=torch.int64, device=dev)

    ref_i8, ref_scale = torch_quant_write_reference(inp, block_ids_t, cut_idx_t, pos_ids_t, token_num, nkvh, dh)
    block_num = ref_i8.shape[0]

    out_i8 = llaisys.Tensor((block_num, token_num, nkvh, dh), dtype=llaisys.DataType.I8, device=llaisys.DeviceType.NVIDIA)
    out_scale = llaisys.Tensor((block_num, token_num, nkvh), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.NVIDIA)
    inp_ll = copy_to_llaisys(inp, llaisys_dtype(dtype_name), llaisys.DeviceType.NVIDIA)
    block_ids_ll = copy_to_llaisys(block_ids_t, llaisys.DataType.I64, llaisys.DeviceType.NVIDIA)
    cut_idx_ll = copy_to_llaisys(cut_idx_t, llaisys.DataType.I64, llaisys.DeviceType.NVIDIA)
    pos_ids_ll = copy_to_llaisys(pos_ids_t, llaisys.DataType.I64, llaisys.DeviceType.NVIDIA)

    llaisys.Ops.kv_cache_move(out_i8, inp_ll, block_ids_ll, cut_idx_ll, pos_ids_ll, out_scale)

    actual_i8 = copy_from_llaisys(out_i8, torch.int8)
    actual_scale = copy_from_llaisys(out_scale, torch.float32)
    assert torch.equal(actual_i8, ref_i8), "kv_cache quant write i8 mismatch"
    assert torch.allclose(actual_scale, ref_scale, atol=1e-6), "kv_cache quant write scale mismatch"


# ── 2. prefill 量化读取 ─────────────────────────────────────

def torch_paged_attention(attn_val, query, k_cache, v_cache, block_ids, scale, totlen=None):
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
        rep = nh // nkvh
        dense_k = dense_k.repeat_interleave(rep, dim=1)
        dense_v = dense_v.repeat_interleave(rep, dim=1)

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


def load_quant_cache(k_cache, v_cache, token_num, nkvh, hd, device_name="nvidia"):
    """FP cache → (llaisys I8 cache, llaisys F32 scale, torch 参考 i8/scale)。"""
    k_i8_t, k_scale_t = torch_quantize_cache(k_cache)
    v_i8_t, v_scale_t = torch_quantize_cache(v_cache)
    block_num = k_cache.shape[0]
    nv = llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU
    k_i8 = llaisys.Tensor((block_num, token_num, nkvh, hd), dtype=llaisys.DataType.I8, device=nv)
    v_i8 = llaisys.Tensor((block_num, token_num, nkvh, hd), dtype=llaisys.DataType.I8, device=nv)
    k_scale = llaisys.Tensor((block_num, token_num, nkvh), dtype=llaisys.DataType.F32, device=nv)
    v_scale = llaisys.Tensor((block_num, token_num, nkvh), dtype=llaisys.DataType.F32, device=nv)
    k_i8.load(c_void_p(k_i8_t.data_ptr()))
    v_i8.load(c_void_p(v_i8_t.data_ptr()))
    k_scale.load(c_void_p(k_scale_t.data_ptr()))
    v_scale.load(c_void_p(v_scale_t.data_ptr()))
    return k_i8, v_i8, k_scale, v_scale


def test_paged_attention_quant(seqlen, token_num, block_num, block_ids_values, nh, nkvh, hd,
                               dtype_name="bf16", atol=2e-2, rtol=2e-2):
    print(f"  [prefill-quant] seqlen={seqlen} block_ids={block_ids_values} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>")
    totlen = len(block_ids_values) * token_num
    q, q_ = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    k_cache, _ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, "nvidia")
    v_cache, _ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, "nvidia")
    attn_val, attn_val_ = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")

    k_i8_, v_i8_, k_scale_, v_scale_ = load_quant_cache(k_cache, v_cache, token_num, nkvh, hd)
    _, block_ids = build_block_ids(block_ids_values)
    scale = 1.0 / (hd ** 0.5)

    cut_idx_t = torch.tensor([0, seqlen], dtype=torch.int64, device=torch_device("nvidia", 0))
    cut_idx_ll = llaisys.Tensor((2,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA)
    cut_idx_ll.load(c_void_p(cut_idx_t.data_ptr()))
    totlen_t = torch.tensor([totlen], dtype=torch.int64, device=torch_device("nvidia", 0))
    totlen_ll = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA)
    totlen_ll.load(c_void_p(totlen_t.data_ptr()))

    torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale)
    llaisys.Ops.paged_attention(attn_val_, q_, k_i8_, v_i8_, block_ids, cut_idx_ll, totlen_ll,
                                (seqlen + 7) // 8, scale, True, None, None, None, k_scale_, v_scale_)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol), "prefill quantized read mismatch"


# ── 3. decode 量化读取 ──────────────────────────────────────

def torch_flash_decoding(attn_val, query, k_cache, v_cache, block_ids, scale, totlen):
    torch_paged_attention(attn_val, query, k_cache, v_cache, block_ids, scale, totlen)


def test_flash_decoding_quant(seqlen, token_num, block_num, block_ids_values, nh, nkvh, hd,
                              dtype_name="bf16", atol=2e-2, rtol=2e-2):
    print(f"  [decode-quant] seqlen={seqlen} block_ids={block_ids_values} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>")
    totlen = len(block_ids_values) * token_num
    assert seqlen == 1
    q, q_ = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    k_cache, _ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, "nvidia")
    v_cache, _ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, "nvidia")
    attn_val, attn_val_ = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")

    k_i8_, v_i8_, k_scale_, v_scale_ = load_quant_cache(k_cache, v_cache, token_num, nkvh, hd)
    _, block_ids = build_block_ids(block_ids_values)
    scale = 1.0 / (hd ** 0.5)

    cut_idx_t = torch.tensor([0, seqlen], dtype=torch.int64, device=torch_device("nvidia", 0))
    cut_idx_ll = llaisys.Tensor((2,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA)
    cut_idx_ll.load(c_void_p(cut_idx_t.data_ptr()))
    totlen_t = torch.tensor([totlen], dtype=torch.int64, device=torch_device("nvidia", 0))
    totlen_ll = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys.DeviceType.NVIDIA)
    totlen_ll.load(c_void_p(totlen_t.data_ptr()))

    buf_num = len(block_ids_values)
    attn_acc = llaisys.Tensor((buf_num, nh, hd), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.NVIDIA)
    attn_sum = llaisys.Tensor((buf_num, nh, 1), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.NVIDIA)
    attn_max = llaisys.Tensor((buf_num, nh, 1), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.NVIDIA)

    torch_flash_decoding(attn_val, q, k_cache, v_cache, block_ids_values, scale, totlen)
    llaisys.Ops.paged_attention(attn_val_, q_, k_i8_, v_i8_, block_ids, cut_idx_ll, totlen_ll,
                                len(block_ids_values), scale, False,
                                attn_acc, attn_sum, attn_max, k_scale_, v_scale_)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol), "decode quantized read mismatch"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for kv_cache_quant test.")

    # 1. 量化写入（三种 FP dtype）
    for dt in ("f32", "f16", "bf16"):
        test_quant_write(dt)

    # 2. prefill 量化读取（Qwen2-1.5B 与 Llama-3.2-3B 的 GQA 比例）
    test_paged_attention_quant(9, 4, 6, [4, 0, 2], nh=8, nkvh=2, hd=128)    # rep=4（通用）
    test_paged_attention_quant(9, 4, 6, [4, 0, 2], nh=12, nkvh=2, hd=128)   # rep=6（Qwen2-1.5B）
    test_paged_attention_quant(9, 4, 6, [4, 0, 2], nh=24, nkvh=8, hd=128)   # rep=3（Llama-3.2-3B）

    # 3. decode 量化读取（flash_decoding 分支，hd=128 且 rep ∈ {6,3}）
    test_flash_decoding_quant(1, 4, 6, [5, 2, 4, 0], nh=12, nkvh=2, hd=128)  # Qwen2-1.5B
    test_flash_decoding_quant(1, 4, 6, [5, 2, 4, 0], nh=24, nkvh=8, hd=128)  # Llama-3.2-3B

    print("\033[92mkv_cache quant tests passed!\033[0m")
