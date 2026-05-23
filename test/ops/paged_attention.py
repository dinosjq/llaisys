import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from ctypes import c_void_p
from test_utils import random_tensor, check_equal, benchmark


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


def tensors_allclose(lhs, rhs, atol=1e-5, rtol=1e-5):
    return torch.allclose(lhs, rhs, atol=atol, rtol=rtol)


def build_block_ids(values, device_name, device_id=0):
    torch_device = torch.device(f"cuda:{device_id}") if device_name == "nvidia" else torch.device("cpu")
    torch_ids = torch.tensor(values, dtype=torch.int32, device=torch_device)
    block_ids = llaisys.Tensor(
        (len(values),),
        dtype=llaisys.DataType.I32,
        device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
        device_id=device_id,
    )
    block_ids.load(c_void_p(torch_ids.data_ptr()))
    return torch_ids, block_ids


def test_op_paged_attention(
    seqlen,
    token_num,
    block_num,
    block_ids_values,
    nh,
    nkvh,
    hd,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="nvidia",
    profile=False,
):
    print(
        f"   seqlen={seqlen} token_num={token_num} block_num={block_num} block_ids={block_ids_values} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>"
    )
    totlen = len(block_ids_values) * token_num
    assert totlen >= seqlen

    device_id = 0

    q, q_ = random_tensor((seqlen, nh, hd), dtype_name, device_name, device_id=device_id)
    k_cache, k_cache_ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, device_name, device_id=device_id)
    v_cache, v_cache_ = random_tensor((block_num, token_num, nkvh, hd), dtype_name, device_name, device_id=device_id)
    attn_val, attn_val_ = random_tensor((seqlen, nh, hd), dtype_name, device_name, device_id=device_id)

    _, block_ids = build_block_ids(block_ids_values, device_name, device_id=device_id)
    scale = 1.0 / (hd**0.5)

    torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale)
    llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, len(block_ids_values), totlen, scale)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol)

    if totlen > seqlen:
        wrong_ref = torch.empty_like(attn_val)
        wrong_out, wrong_out_ = random_tensor((seqlen, nh, hd), dtype_name, device_name, device_id=device_id)
        torch_paged_attention(wrong_ref, q, k_cache, v_cache, block_ids_values, scale, totlen=seqlen)
        llaisys.Ops.paged_attention(wrong_out_, q_, k_cache_, v_cache_, block_ids, len(block_ids_values), seqlen, scale)
        assert check_equal(wrong_out_, wrong_ref, atol=atol, rtol=rtol)
        assert not tensors_allclose(wrong_out, attn_val, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale),
            lambda: llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, len(block_ids_values), totlen, scale),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    testShapes = [
        # seqlen, token_num, block_num, block_ids, nh, nkvh, hd
        (5, 4, 5, [4, 1, 3], 4, 2, 8),
        (9, 4, 6, [5, 2, 4, 0], 4, 1, 8),
        (1, 8, 10, [9, 3, 7, 1], 8, 2, 16),
        (7, 4, 12, [11, 6, 2, 9, 0], 12, 3, 8),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.paged_attention on {args.device}")
    for shape in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_paged_attention(*shape, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
