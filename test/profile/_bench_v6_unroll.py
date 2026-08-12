#!/usr/bin/env python3
"""v6 no-unroll vs unroll benchmark."""
import torch, sys, os, ctypes, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5

configs = [(1,256),(1,2048),(4,2048),(8,1024),(16,512),(32,256)]
script = os.path.join(os.path.dirname(__file__), '_run_v6unroll.py')
with open(script, 'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}
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
    for ur in [1, 0]:
        args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
              qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
              TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,6,16,1,2,ur)
        for _ in range(3): lib.llaisysFlashDecodingV6(*args)
        torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(f"V6U{{ur}}_B{{batch}}T{{totlen}}")
        lib.llaisysFlashDecodingV6(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running nsys...")
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v6ur","--force-overwrite=true",
                "python3",script], check=False, timeout=600, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v6ur_t_cuda_gpu_trace.csv","/tmp/v6ur.sqlite"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v6ur_t","/tmp/v6ur.nsys-rep"], check=True, timeout=120, capture_output=True)

kernels = []
with open("/tmp/v6ur_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if 'flash_decoding_v6' in n:
            kernels.append((d, n))
print(f"Total: {len(kernels)}")
per = 16  # ur=1 (8) + ur=0 (8)
print(f"\n{'b×totlen':>14} {'UR1(us)':>8} {'UR0(us)':>8} {'UR0/UR1':>8}")
for ci,(b,t) in enumerate(configs):
    base=ci*16
    ur1=kernels[base+6][0]/1e3+kernels[base+7][0]/1e3
    ur0=kernels[base+14][0]/1e3+kernels[base+15][0]/1e3
    print(f"  {b:2d}×{t:5d}  {ur1:6.1f}  {ur0:6.1f}  {ur0/ur1:.2f}x")
