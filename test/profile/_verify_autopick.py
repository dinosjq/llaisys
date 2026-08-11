#!/usr/bin/env python3
"""验证启发式自动选参 (heads=0, tile_k=0) vs 每个 config 的实际最优."""
import time, torch, sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV2.argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p,
    c_size_t, c_size_t, c_size_t, c_size_t,
    c_size_t, c_int, c_float,
    c_size_t, c_size_t, c_size_t, c_size_t,
    c_int, c_int,
]

_cfg_cache = {}
def run_v2(batch, totlen, heads, tile_k):
    """cudaEvent 批量计时（N 次累积取平均，排除 CPU launch 开销）."""
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
    N = 100
    for _ in range(20):
        lib.llaisysFlashDecodingV2(*args)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(N):
        lib.llaisysFlashDecodingV2(*args)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / N * 1000.0  # us per call

BATCHES = [1, 2, 4, 8, 16, 32]
TOTLENS = [256, 512, 1024, 2048]
PARAMS = [(h, t) for h in [1, 2, 3, 6] for t in [4, 8, 16]]

print("=== 自动选参 vs 最优 ===")
print(f"{'batch':>5} {'totlen':>6} {'auto(us)':>9} {'best(us)':>9} {'loss':>6}")
total_loss = 0.0; n = 0
for batch in BATCHES:
    for totlen in TOTLENS:
        best_lat = min(run_v2(batch, totlen, h, t) for h, t in PARAMS)
        auto_lat = run_v2(batch, totlen, 0, 0)   # 自动选参
        loss = (auto_lat - best_lat) / best_lat * 100
        total_loss += loss; n += 1
        flag = "" if loss < 5 else "  <-- 损失>5%"
        print(f"{batch:5d} {totlen:6d} {auto_lat:9.1f} {best_lat:9.1f} {loss:5.1f}%{flag}")
print(f"\n平均损失: {total_loss/n:.1f}%")
