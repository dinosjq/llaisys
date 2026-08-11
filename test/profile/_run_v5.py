#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'/home/songjq/llaisys-main/test/profile/..')
sys.path.insert(0, r'/home/songjq/llaisys-main/test/profile/../../python')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = [(1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096), (4, 2048), (8, 1024), (16, 512), (32, 256), (2, 256), (2, 2048), (4, 1024), (8, 512), (16, 256)]; tests = [(6, 8), (2, 16)]
for (batch, totlen) in configs:
    bn=totlen//TN; mb=bn+1; tb=batch*bn
    tbids=torch.zeros((batch,mb),dtype=torch.int64,device=DEV)
    for b in range(batch):
        tbids[b,0]=(b+1)*bn; tbids[b,1:]=torch.arange(b*bn,(b+1)*bn,device=DEV)
    qt=torch.randn((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    kt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    vt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    cut=torch.arange(batch+1,dtype=torch.int64,device=DEV)
    tot=torch.full((batch,),totlen,dtype=torch.int64,device=DEV)
    s=1.0/(HD**0.5)
    for (h,tk) in tests:
        for fn_name in ['llaisysFlashDecodingV3','llaisysFlashDecodingV5']:
            av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
            acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
            asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
                  qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
                  TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,0)
            getattr(lib,fn_name)(*args); torch.cuda.synchronize()
            label=f"{fn_name[-2:]}_B{batch}T{totlen}_H{h}K{tk}"
            torch.cuda.nvtx.range_push(label)
            getattr(lib,fn_name)(*args); torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
print("done")
