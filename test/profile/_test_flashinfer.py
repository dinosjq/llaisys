#!/usr/bin/env python3
"""Test FlashInfer decode with JIT compilation."""
import os, sys
os.environ["FLASHINFER_NVCC_APPEND_FLAGS"] = "-allow-unsupported-compiler -ccbin /usr/bin/gcc-12"
import torch
import flashinfer

NH, NKVH, HD, TN = 12, 2, 128, 64
batch, totlen = 1, 256
bn = totlen // TN

kv_data = torch.randn(bn, 2, TN, NKVH, HD, dtype=torch.bfloat16, device='cuda:0')
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    torch.empty(0, dtype=torch.bfloat16, device='cuda:0'), 'NHD')
kv_indptr = torch.tensor([0, bn], dtype=torch.int32, device='cuda:0')
kv_indices = torch.arange(bn, dtype=torch.int32, device='cuda:0')
kv_last_page_len = torch.tensor([TN], dtype=torch.int32, device='cuda:0')

wrapper.plan(kv_indptr, kv_indices, kv_last_page_len, NH, NKVH, HD, TN,
             q_data_type=torch.bfloat16, kv_data_type=torch.bfloat16)

q = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device='cuda:0')
out = wrapper.run(q, kv_data)
print(f"FlashInfer OK: shape={out.shape}, mean={out.float().mean().item():.4f}")
