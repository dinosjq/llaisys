#!/usr/bin/env python3
"""ncu target for FlashInfer NH=12 group_size=6 kernel."""
import os, sys
os.environ["FLASHINFER_NVCC_APPEND_FLAGS"] = "-allow-unsupported-compiler -ccbin /usr/bin/gcc-12"
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
fi_ws = torch.empty(128*1024*1024, dtype=torch.float32, device="cuda")

for batch, totlen in [(16, 512), (4, 2048)]:
    bn = totlen // TN
    indptr = torch.arange(0, batch+1, dtype=torch.int32, device=DEV) * bn
    indices = torch.tile(torch.arange(bn, dtype=torch.int32, device=DEV), (batch,))
    last_page = torch.full((batch,), TN, dtype=torch.int32, device=DEV)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(fi_ws, kv_layout="NHD")
    wrapper.plan(indptr, indices, last_page, NH, NKVH, HD, TN, q_data_type=torch.bfloat16)
    q = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device=DEV)
    kv = torch.randn(batch*bn, TN, NKVH, HD, dtype=torch.bfloat16, device=DEV)
    for _ in range(5):
        wrapper.run(q, (kv, kv))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"FI_B{batch}T{totlen}")
    wrapper.run(q, (kv, kv))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
