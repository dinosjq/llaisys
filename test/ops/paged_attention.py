import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from ctypes import c_void_p
from test_utils import random_tensor, check_equal, benchmark, torch_device, llaisys_device, llaisys_dtype


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
    """
    Build V3-format block_ids tensor: first element = block count, then block IDs.
    Format matches qwen2.cpp prepare_block_table: [count, id_0, id_1, ...].
    """
    torch_device = torch.device(f"cuda:{device_id}") if device_name == "nvidia" else torch.device("cpu")
    count = len(values)
    padded = [count] + list(values)
    torch_ids = torch.tensor(padded, dtype=torch.int64, device=torch_device)
    block_ids = llaisys.Tensor(
        (1, len(padded)),  # 2D: (batch=1, count + num_blocks)
        dtype=llaisys.DataType.I64,
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

    # construct cut_idx and tot_len tensors for single-sample new-signature call
    from ctypes import c_void_p
    cut_idx_t = torch.tensor([0, seqlen], dtype=torch.int64, device=torch_device(device_name, device_id))
    cut_idx_ll = llaisys.Tensor((2,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
    cut_idx_ll.load(c_void_p(cut_idx_t.data_ptr()))

    totlen_t = torch.tensor([totlen], dtype=torch.int64, device=torch_device(device_name, device_id))
    totlen_ll = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
    totlen_ll.load(c_void_p(totlen_t.data_ptr()))

    torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale)
    # prefill 路径 tot_task_num = Σceil(seq_len/BLOCK_M)，BLOCK_M=8
    llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, cut_idx_ll, totlen_ll,
                                (seqlen + 7) // 8, scale)
    assert check_equal(attn_val_, attn_val, atol=atol, rtol=rtol)

    if totlen > seqlen:
        wrong_ref = torch.empty_like(attn_val)
        wrong_out, wrong_out_ = random_tensor((seqlen, nh, hd), dtype_name, device_name, device_id=device_id)
        torch_paged_attention(wrong_ref, q, k_cache, v_cache, block_ids_values, scale, totlen=seqlen)

        # construct cut_idx/totlen for this shorter totlen
        cut_idx_t2 = torch.tensor([0, seqlen], dtype=torch.int64, device=torch_device(device_name, device_id))
        cut_idx_ll2 = llaisys.Tensor((2,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
        cut_idx_ll2.load(c_void_p(cut_idx_t2.data_ptr()))

        totlen_t2 = torch.tensor([seqlen], dtype=torch.int64, device=torch_device(device_name, device_id))
        totlen_ll2 = llaisys.Tensor((1,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
        totlen_ll2.load(c_void_p(totlen_t2.data_ptr()))

        llaisys.Ops.paged_attention(wrong_out_, q_, k_cache_, v_cache_, block_ids, cut_idx_ll2, totlen_ll2,
                                    (seqlen + 7) // 8, scale)
        assert check_equal(wrong_out_, wrong_ref, atol=atol, rtol=rtol)
        assert not tensors_allclose(wrong_out, attn_val, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale),
            lambda: llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, cut_idx_ll, totlen_ll,
                                                (seqlen + 7) // 8, scale),
            device_name,
        )


def test_op_paged_attention_batched(
    batch_cases,
    token_num,
    nh,
    nkvh,
    hd,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="nvidia",
    profile=False,
):
    device_id = 0
    batch = len(batch_cases)
    seqlens = [case[0] for case in batch_cases]
    block_nums = [case[1] for case in batch_cases]
    block_ids_list = [case[2] for case in batch_cases]
    max_block_num = max(block_nums)
    totlen_per_sample = [len(bids) * token_num for bids in block_ids_list]

    for seqlen, block_num, bids in batch_cases:
        print(f"   seqlen={seqlen} token_num={token_num} block_num={block_num} block_ids={bids} nh={nh} nkvh={nkvh} hd={hd} dtype <{dtype_name}>")

    q_torch_list = []
    for seqlen in seqlens:
        q_sample_torch, _ = random_tensor(
            (seqlen, nh, hd), dtype_name, device_name, device_id=device_id
        )
        q_torch_list.append(q_sample_torch)

    q_cat = torch.cat(q_torch_list, dim=0)
    _, q_cat_ll = random_tensor(
        q_cat.shape, dtype_name, device_name, device_id=device_id
    )
    api = llaisys.RuntimeAPI(llaisys_device(device_name))
    api.memcpy_sync(
        q_cat_ll.data_ptr(),
        q_cat.data_ptr(),
        q_cat.numel() * q_cat.element_size(),
        llaisys.MemcpyKind.D2D,
    )

    k_cache, k_cache_ = random_tensor((max_block_num, token_num, nkvh, hd), dtype_name, device_name, device_id=device_id)
    v_cache, v_cache_ = random_tensor((max_block_num, token_num, nkvh, hd), dtype_name, device_name, device_id=device_id)

    mb = max(len(bids) for bids in block_ids_list) + 1
    torch_block_ids = []
    for bids in block_ids_list:
        row = [len(bids)] + list(bids) + [0] * (mb - len(bids) - 1)
        torch_block_ids.append(row)
    torch_block_ids = torch.tensor(torch_block_ids, dtype=torch.int64, device=torch_device(device_name, device_id))
    block_ids_ll = llaisys.Tensor((batch, mb), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
    block_ids_ll.load(c_void_p(torch_block_ids.data_ptr()))

    cut_idx = [0]
    for s in seqlens:
        cut_idx.append(cut_idx[-1] + s)
    cut_idx_torch = torch.tensor(cut_idx, dtype=torch.int64, device=torch_device(device_name, device_id))
    cut_idx_ll = llaisys.Tensor((len(cut_idx),), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
    cut_idx_ll.load(c_void_p(cut_idx_torch.data_ptr()))

    totlen_torch = torch.tensor(totlen_per_sample, dtype=torch.int64, device=torch_device(device_name, device_id))
    totlen_ll = llaisys.Tensor((batch,), dtype=llaisys.DataType.I64, device=llaisys_device(device_name), device_id=device_id)
    totlen_ll.load(c_void_p(totlen_torch.data_ptr()))

    tot_seqlen = sum(seqlens)
    attn_val_ll = llaisys.Tensor((tot_seqlen, nh, hd), dtype=llaisys_dtype(dtype_name), device=llaisys_device(device_name), device_id=device_id)
    scale = 1.0 / (hd ** 0.5)

    # prefill: tot_task_num = Σceil(seq_len/BLOCK_M)，BLOCK_M=8
    tot_task_num = sum((s + 7) // 8 for s in seqlens)
    llaisys.Ops.paged_attention(attn_val_ll, q_cat_ll, k_cache_, v_cache_, block_ids_ll, cut_idx_ll, totlen_ll, tot_task_num, scale)

    reference = torch.empty_like(q_cat)
    offset = 0
    for seqlen, block_ids_values, totlen in zip(
        seqlens, block_ids_list, totlen_per_sample
    ):
        torch_paged_attention(
            reference[offset:offset + seqlen],
            q_cat[offset:offset + seqlen],
            k_cache,
            v_cache,
            block_ids_values,
            scale,
            totlen=totlen,
        )
        offset += seqlen

    attn_val = torch.empty_like(reference)
    api.memcpy_sync(
        attn_val.data_ptr(),
        attn_val_ll.data_ptr(),
        attn_val.numel() * attn_val.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    assert torch.allclose(attn_val, reference, atol=atol, rtol=rtol)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    test_shapes = [
        # seqlen, token_num, block_num, block_ids, nh, nkvh, hd
        (5, 4, 5, [4, 1, 3], 4, 2, 32),
        (9, 4, 6, [5, 2, 4, 0], 4, 1, 64),
        (1, 8, 10, [9, 3, 7, 1], 8, 2, 96),
        (7, 4, 12, [11, 6, 2, 9, 0], 12, 3, 128),
    ]
    test_dtype_prec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.paged_attention on {args.device}")
    for shape in test_shapes:
        for dtype_name, atol, rtol in test_dtype_prec:
            seqlen, token_num, block_num, block_ids, nh, nkvh, hd = shape
            batch_cases = [(seqlen, block_num, block_ids)]
            test_op_paged_attention_batched(
                batch_cases,
                token_num,
                nh,
                nkvh,
                hd,
                dtype_name,
                atol,
                rtol,
                args.device,
                args.profile,
            )

    mixed_batch_cases = [
        (5, 6, [5, 1, 3]),
        (9, 6, [4, 0, 2]),
    ]
    for dtype_name, atol, rtol in test_dtype_prec:
        test_op_paged_attention_batched(
            mixed_batch_cases,
            4,
            8,
            2,
            128,
            dtype_name,
            atol,
            rtol,
            args.device,
            args.profile,
        )

    print("\033[92mTest passed!\033[0m\n")
