#!/usr/bin/env python3
"""实验对比: v3 全体参与 vs producer/consumer 分离 × token_num 变长.
nsys 精确 GPU 计时，每个组合 1 帧避谈预热噪声."""
import torch, sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD = 12, 2, 128
DEV = torch.device("cuda:0")
BV2 = 2  # bf16

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
# V3: last 3 params = heads, tile_k, use_prodcon
lib.llaisysFlashDecodingV3.argtypes = [
    c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p, c_void_p,
    c_void_p, c_void_p, c_void_p, c_size_t, c_size_t, c_size_t, c_size_t,
    c_size_t, c_int, c_float, c_size_t, c_size_t, c_size_t, c_size_t, c_int, c_int, c_int,
]
# Ex: same signature but with flags instead of use_prodcon
lib.llaisysFlashDecodingEx.argtypes = lib.llaisysFlashDecodingV3.argtypes

def once(lib, fn, batch, totlen, token_num, heads, tile_k, use_prodcon, label):
    bn = totlen // token_num; mb = bn + 1
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b + 1) * bn; tbids[b, 1:] = torch.arange(bn, device=DEV)
    qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    kt = torch.randn((bn, token_num, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    vt = torch.randn((bn, token_num, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    cut = torch.arange(batch + 1, dtype=torch.int64, device=DEV)
    tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
    tb = batch * bn
    acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
    asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    s = 1.0 / (HD ** 0.5)
    # V3: (heads, tile_k, use_prodcon)
    # Ex: (heads, tile_k, flags) where flags & 1 = use_prodcon
    args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), tbids.data_ptr(), cut.data_ptr(), tot.data_ptr(),
            token_num, batch, mb, tb, 1, BV2, s, NH, HD, HD, NKVH, heads, tile_k, use_prodcon)
    getattr(lib, fn)(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    getattr(lib, fn)(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

# ── 实验1: v3 全体 vs producer/consumer @16×2048 H6T8 ──────────
print("=== 实验1: v3全体 vs prod/con @16x2048 H6T8 ===")
# v3 文件只有全体参与；experiments 文件有 prodcon
once(lib, "llaisysFlashDecodingV3", 16, 2048, 64, 6, 8, 0, "v3_all")
once(lib, "llaisysFlashDecodingEx", 16, 2048, 64, 6, 8, 1, "v3_prodcon")

# ── 实验2: token_num 变长 @1×2048 H6T8 ─────────────────────────
print("=== 实验2: token_num 变长 @1x2048 H6T8 ===")
for tn in [64, 128, 256]:
    once(lib, "llaisysFlashDecodingV3", 1, 2048, tn, 6, 8, 0, f"TN{tn}")

# ── 实验3: token_num 变长 @16×2048 H6T8 ────────────────────────
for tn in [64, 128, 256]:
    once(lib, "llaisysFlashDecodingV3", 16, 2048, tn, 6, 8, 0, f"b16_TN{tn}")
