#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'/home/songjq/llaisys-main/test/profile/..')
sys.path.insert(0, r'/home/songjq/llaisys-main/test/profile/../../python')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
for fn in ['llaisysFlashDecodingV3','llaisysFlashDecodingV5','llaisysFlashDecodingV6']:
    getattr(lib,fn).argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = [(1, 256), (1, 512), (1, 1024), (1, 2048), (1, 4096), (2, 256), (2, 2048), (4, 512), (4, 1024), (4, 2048), (8, 256), (8, 512), (8, 1024), (16, 256), (16, 512), (32, 256)]; v3_tests = [(6, 8, 0), (2, 16, 0)]; vv_tests = [(6, 8, 1), (6, 8, 4), (3, 16, 4), (2, 16, 1), (3, 16, 8)]
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
    for (h,tk,co) in v3_tests:
        av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
        acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
        asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
        amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
        args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
              qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
              TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,co)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        label=f"V3_B{batch}T{totlen}_H{h}K{tk}"
        torch.cuda.nvtx.range_push(label)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    for (h,tk,co) in vv_tests:
        for fn in ['llaisysFlashDecodingV5','llaisysFlashDecodingV6']:
            av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
            acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
            asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
            args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
                  qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
                  TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,co)
            getattr(lib,fn)(*args); torch.cuda.synchronize()
            label=f"{fn[-2:]}_B{batch}T{totlen}_H{h}K{tk}C{co}"
            torch.cuda.nvtx.range_push(label)
            getattr(lib,fn)(*args); torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
print("done")
