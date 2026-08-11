#!/usr/bin/env python3
"""CO=1 vs CO=4 ncu profiling for batch=16, totlen=2048."""
import torch, sys, os, ctypes
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

batch, totlen = 16, 2048
bn = totlen // TN; mb = bn + 1; tb = batch * bn
tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
for b in range(batch):
    tbids[b, 0] = (b+1)*bn
    tbids[b, 1:] = torch.arange(b*bn, (b+1)*bn, device=DEV)
qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
kt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
vt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
cut = torch.arange(batch+1, dtype=torch.int64, device=DEV)
tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
s = 1.0/(HD**0.5)

def run(co, label):
    av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
    asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    args = (av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
            qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
            TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,6,8,co)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

run(1, "CO1_H6T8")
run(4, "CO4_H6T8")
print("done")
