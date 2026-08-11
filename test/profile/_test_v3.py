#!/usr/bin/env python3
"""v3 (cp.async 流水线) 正确性 + 性能 vs v2."""
import time, torch, sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2  # bf16

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_size_t, c_size_t, c_size_t, c_size_t,
    c_size_t, c_int, c_float,
    c_size_t, c_size_t, c_size_t, c_size_t,
    c_int, c_int,
]

_cfg_cache = {}
def run_v3(batch, totlen, heads, tile_k):
    key = (batch, totlen)
    if key not in _cfg_cache:
        bn = totlen // TN; mb = bn + 1
        tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
        for b in range(batch):
            tbids[b, 0] = (b + 1) * bn; tbids[b, 1:] = torch.arange(bn, device=DEV)
        qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
        kt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
        vt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
        av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
        cut = torch.arange(batch + 1, dtype=torch.int64, device=DEV)
        tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
        tb = batch * bn
        acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
        asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        scale = 1.0 / (HD ** 0.5)
        _cfg_cache[key] = (av, acc, asum, amax, qt, kt, vt, tbids, cut, tot, tb, scale)
    av, acc, asum, amax, qt, kt, vt, bids, cut, tot, tb, scale = _cfg_cache[key]
    args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), bids.data_ptr(), cut.data_ptr(), tot.data_ptr(),
            TN, batch, totlen//TN+1, tb, 1, BV2, scale, NH, HD, HD, NKVH, heads, tile_k)
    for _ in range(10):
        lib.llaisysFlashDecodingV3(*args)
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lib.llaisysFlashDecodingV3(*args)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e6)
    return best, av, qt, kt, vt, bids, totlen

def check(av, qt, kt, vt, bids, totlen):
    dk = torch.empty((totlen, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    dv = torch.empty((totlen, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    for tid in range(totlen):
        dk[tid] = kt[int(bids[0, 1 + tid // TN]), tid % TN]
        dv[tid] = vt[int(bids[0, 1 + tid // TN]), tid % TN]
    rep = NH // NKVH
    dk2 = dk.repeat_interleave(rep, 1); dv2 = dv.repeat_interleave(rep, 1)
    qr = qt.transpose(0, 1); kr = dk2.transpose(0, 1); vr = dv2.transpose(0, 1)
    scale = 1.0 / (HD ** 0.5)
    aw = torch.softmax(torch.matmul(qr, kr.transpose(-2, -1)) * scale, dim=-1)
    ref = torch.matmul(aw, vr).transpose(0, 1)
    from llaisys.runtime import RuntimeAPI
    api = RuntimeAPI(llaisys.libllaisys.DeviceType.NVIDIA)
    out = torch.empty_like(qt)
    api.memcpy_sync(out.data_ptr(), av.data_ptr(), out.numel() * out.element_size(), 2)
    return (out.float() - ref.float()).abs().max().item()

# Correctness across configs
print("=== Correctness (v3 cp.async) ===")
for batch, totlen, h, t in [(1, 256, 2, 16), (1, 2048, 6, 8), (4, 2048, 6, 8), (16, 2048, 6, 8)]:
    lat, av, qt, kt, vt, bids, tl = run_v3(batch, totlen, h, t)
    d = check(av, qt, kt, vt, bids, tl)
    print(f"  {batch}x{totlen} H{h}T{t}: max_diff={d:.5f} {'PASS' if d < 0.02 else 'FAIL'}")

# Benchmark v3 (auto-pick)
print("\n=== Latency v3 (auto-pick) ===")
print(f"{'batch':>5} {'totlen':>6} {'v3(us)':>9}")
for batch, totlen in [(1,256),(1,1024),(1,2048),(4,2048),(8,1024),(8,2048),(16,2048),(32,256)]:
    lat, *_ = run_v3(batch, totlen, 0, 0)   # auto
    print(f"{batch:5d} {totlen:6d} {lat:9.1f}")
