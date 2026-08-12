#!/usr/bin/env python3
"""统一测量 v3/v4/v5/v6 各版本最优参数, 同 nsys 方法同 config 集.
v1/v2 从原文档引用."""
import torch, sys, os, ctypes, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5

configs = [(1,256),(1,1024),(1,2048),(1,4096),(4,512),(4,2048),(8,1024),(16,512),(32,256)]
# 各版本最优参数 (V3: use_prodcon=0; V4/V5: H,TK,CO; V6: H,TK,CO,NS,UR)
versions = [
    ("V3", "llaisysFlashDecodingV3", (6,8,0)),
    ("V4", "llaisysFlashDecodingV4", (6,8,1)),
    ("V5", "llaisysFlashDecodingV5", (6,8,1)),
    ("V6", "llaisysFlashDecodingV6", (6,16,1,2,1)),
]

script = os.path.join(os.path.dirname(__file__), '_run_evol.py')
with open(script, 'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; versions = {repr([(v[0], v[2]) for v in versions])}
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
    av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
    asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    for (tag, params) in versions:
        fn = "llaisysFlashDecoding" + tag
        args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
              qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
              TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH)+params
        for _ in range(3): getattr(lib,fn)(*args)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"{{tag}}_B{{batch}}T{{totlen}}")
        getattr(lib,fn)(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running nsys...")
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/evol","--force-overwrite=true",
                "python3",script], check=False, timeout=900, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/evol_t_cuda_gpu_trace.csv","/tmp/evol.sqlite"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/evol_t","/tmp/evol.nsys-rep"], check=True, timeout=120, capture_output=True)

import re
kernels = []
with open("/tmp/evol_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if re.search(r'flash_decoding_v[3456]_kernel|flash_decoding_v[3456]_reduce', n):
            kernels.append((d, n))
print(f"Total: {len(kernels)}")
per = 4 * 8  # 4 versions × (3w+1m) × 2 kernel
print(f"\n{'b×totlen':>14} {'v3(us)':>8} {'v4(us)':>8} {'v5(us)':>8} {'v6(us)':>8} {'v6/v3':>7}")
for ci,(b,t) in enumerate(configs):
    base=ci*32
    vals = []
    for vi in range(4):
        vals.append(kernels[base+vi*8+6][0]/1e3 + kernels[base+vi*8+7][0]/1e3)
    print(f"  {b:2d}×{t:5d}  {vals[0]:6.1f}  {vals[1]:6.1f}  {vals[2]:6.1f}  {vals[3]:6.1f}  {vals[3]/vals[0]:.2f}x")
