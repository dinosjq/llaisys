#!/usr/bin/env python3
"""参数扫描: 优化A(HEADS=nh/grid.y) × 优化B(TILE_K 双缓冲粒度)。

扫描模式:
  A-only : TILE_K=4 (v1值), heads ∈ {1,2,3,6}
  B-only : heads=1 (v1值), TILE_K ∈ {4,8,16,32}
  combo  : heads ∈ {1,2,3,6} × TILE_K ∈ {4,8,16,32}
"""
import time, torch, sys, os, ctypes, itertools
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2  # bf16

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

def run_v2(batch, totlen, heads, tile_k):
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
    args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), tbids.data_ptr(), cut.data_ptr(), tot.data_ptr(),
            TN, batch, mb, tb, 1, BV2, scale, NH, HD, HD, NKVH, heads, tile_k)
    for _ in range(10):
        lib.llaisysFlashDecodingV2(*args)
    torch.cuda.synchronize()
    best = 1e18
    for _ in range(30):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lib.llaisysFlashDecodingV2(*args)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1e6)
    return best, av, qt, kt, vt, tbids, totlen

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

# correctness for all head combos at 1x256
print("=== Correctness (batch=1, totlen=256) ===")
for heads in [1, 2, 3, 6]:
    for tk in [4, 8, 16, 32]:
        lat, av, qt, kt, vt, bids, tl = run_v2(1, 256, heads, tk)
        d = check(av, qt, kt, vt, bids, tl)
        ok = "PASS" if d < 0.02 else "FAIL"
        print(f"  heads={heads} tile_k={tk:2d}: max_diff={d:.5f} {ok}")

# configs to sweep
SWEEP = [(1, 2048), (8, 1024), (16, 2048), (32, 256)]

def sweep_table(label, configs):
    print(f"\n=== {label} (us) ===")
    for batch, totlen in SWEEP:
        row = f"  {batch}x{totlen}:"
        for heads, tk in configs:
            lat, *_ = run_v2(batch, totlen, heads, tk)
            row += f"  H{heads}/T{tk}={lat:6.1f}"
        print(row)

# A-only: TILE_K=4 (v1), vary heads
sweep_table("优化A单独 (TILE_K=4, HEADS扫描)", [(h, 4) for h in [1, 2, 3, 6]])
# B-only: HEADS=1 (v1), vary tile_k
sweep_table("优化B单独 (HEADS=1, TILE_K扫描)", [(1, t) for t in [4, 8, 16, 32]])
# combo
sweep_table("组合 (HEADS × TILE_K)", [(h, t) for h in [1, 2, 3, 6] for t in [4, 8, 16, 32]])

# summary: best per config
print("\n=== 最优参数 per config ===")
for batch, totlen in SWEEP:
    best = None
    for heads, tk in itertools.product([1,2,3,6], [4,8,16,32]):
        lat, *_ = run_v2(batch, totlen, heads, tk)
        if best is None or lat < best[0]:
            best = (lat, heads, tk)
    print(f"  {batch}x{totlen}: best={best[0]:.1f}us at heads={best[1]} tile_k={best[2]}")
