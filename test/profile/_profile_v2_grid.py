#!/usr/bin/env python3
"""nsys profile 目标: 每个 (config, params) 组合发 1 个 kernel，nvtx range 标记。
配合 `nsys profile -t cuda,nvtx` + cuda_gpu_trace 解析。
"""
import torch, sys, os, ctypes
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

def profile_once(batch, totlen, heads, tile_k, label):
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
    # 预热 + 触发该参数的模板实例化
    lib.llaisysFlashDecodingV2(*args)
    torch.cuda.synchronize()
    # nvtx 标记的正式测量
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV2(*args)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

CONFIGS = [(b, t) for b in [1, 2, 4, 8, 16, 32] for t in [256, 512, 1024, 2048]]
CANDIDATES = [(h, t, f"H{h}T{t}") for h in [1, 2, 3, 6] for t in [4, 8, 16]]

for batch, totlen in CONFIGS:
    # 先 auto (heads=0, tile_k=0)
    profile_once(batch, totlen, 0, 0, f"c{batch}x{totlen}_AUTO")
    # 再 12 个候选
    for h, t, lbl in CANDIDATES:
        profile_once(batch, totlen, h, t, f"c{batch}x{totlen}_{lbl}")
