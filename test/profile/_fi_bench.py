#!/usr/bin/env python3
"""FlashInfer vs LLAISYS v4 coalesce comparison. Uses NH=8,NKVH=2,HD=128 (group_size=4) for FlashInfer compatibility."""
import torch, sys, os, ctypes, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper

# GQA config: FlashInfer requires group_size in {1,2,4,8}
NH, NKVH, HD, TN = 8, 2, 128, 64
DEV = torch.device("cuda:0")

# FlashInfer setup
fi_workspace = torch.empty(128*1024*1024, dtype=torch.float32, device="cuda")

# LLAISYS v4 C wrapper
BV2 = 2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

print(f"NH={NH} NKVH={NKVH} HD={HD} TN={TN}  group_size={NH//NKVH}")
print(f"{'batch':>5} {'totlen':>6} | {'FI_par':>10} {'FI_total':>10} | {'v4_par':>10} {'v4_total':>10} {'best(H,TK,CO)':>14} | {'FI/v4':>7}")

for batch, totlen in [(1,2048),(4,2048),(8,2048),(16,2048),(1,4096),(4,4096),(16,4096)]:
    bn = totlen // TN; tb = batch * bn; mb = bn + 1

    # === FlashInfer ===
    indptr = torch.arange(0, batch+1, dtype=torch.int32, device=DEV) * bn
    indices = torch.tile(torch.arange(bn, dtype=torch.int32, device=DEV), (batch,))
    last_page = torch.full((batch,), TN, dtype=torch.int32, device=DEV)
    wrapper = BatchDecodeWithPagedKVCacheWrapper(fi_workspace, kv_layout="NHD")
    wrapper.plan(indptr, indices, last_page, NH, NKVH, HD, TN, q_data_type=torch.bfloat16)

    q_fi = torch.randn(batch, NH, HD, dtype=torch.bfloat16, device=DEV)
    k_fi = torch.randn(tb, TN, NKVH, HD, dtype=torch.bfloat16, device=DEV)
    v_fi = torch.randn_like(k_fi)

    for _ in range(3):  # warmup
        wrapper.run(q_fi, (k_fi, v_fi))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("FI")
    wrapper.run(q_fi, (k_fi, v_fi))
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    # === LLAISYS v4 ===
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b+1)*bn
        tbids[b, 1:] = torch.arange(b*bn, (b+1)*bn, device=DEV)
    cut = torch.arange(batch+1, dtype=torch.int64, device=DEV)
    tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
    s = 1.0/(HD**0.5)
    q_ll = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    kt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    vt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)

    # Test v4 with best H,TK,CO combos
    best_total = 1e9; best_cfg = None
    for h,tk,co in [(6,8,1),(6,8,2),(6,8,4),(6,8,8),(3,16,1),(3,16,8),(2,16,1)]:
        av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
        acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
        asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        args = (av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
                q_ll.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
                TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,co)
        lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"LL_H{h}T{tk}_C{co}")
        lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()

print("done")
