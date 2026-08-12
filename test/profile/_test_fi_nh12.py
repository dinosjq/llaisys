#!/usr/bin/env python3
"""Test FlashInfer with NH=12, NKVH=2 (group_size=6) after patch."""
import os, sys
os.environ["FLASHINFER_NVCC_APPEND_FLAGS"] = "-allow-unsupported-compiler -ccbin /usr/bin/gcc-12"
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

NH, NKVH, HD, TN = 12, 2, 128, 64
workspace = torch.empty(128*1024*1024, dtype=torch.float32, device="cuda")

for batch, totlen in [(1, 256), (4, 2048)]:
    bn = totlen // TN
    indptr = torch.arange(0, batch+1, dtype=torch.int32, device="cuda") * bn
    indices = torch.tile(torch.arange(bn, dtype=torch.int32, device="cuda"), (batch,))
    last_page = torch.full((batch,), TN, dtype=torch.int32, device="cuda")
    wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last_page, NH, NKVH, HD, TN, q_data_type=torch.bfloat16)
    q = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device="cuda")
    kv = torch.randn(batch*bn, TN, NKVH, HD, dtype=torch.bfloat16, device="cuda")
    out = wrapper.run(q, (kv, kv))
    print(f"NH=12 batch={batch} totlen={totlen}: OK shape={out.shape} finite={torch.isfinite(out).all().item()}")
print("done")
