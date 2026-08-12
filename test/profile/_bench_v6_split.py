#!/usr/bin/env python3
"""v6 H6T16C1S2U1: parallel vs reduce kernel 耗时拆分."""
import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5

configs = [(1,256),(1,2048),(1,4096),(4,2048),(8,1024),(16,512),(32,256)]
script = os.path.join(os.path.dirname(__file__), '_run_v6split.py')
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
    args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
          qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
          TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,6,16,1,2,1)
    for _ in range(3): lib.llaisysFlashDecodingV6(*args)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"B{{batch}}T{{totlen}}")
    lib.llaisysFlashDecodingV6(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running nsys...")
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v6sp","--force-overwrite=true",
                "python3",script], check=False, timeout=600, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v6sp_t_cuda_gpu_trace.csv","/tmp/v6sp.sqlite"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v6sp_t","/tmp/v6sp.nsys-rep"], check=True, timeout=120, capture_output=True)

par_pat = re.compile(r"flash_decoding_v6_kernel")
red_pat = re.compile(r"flash_decoding_v6_reduce")
pars, reds = [], []
with open("/tmp/v6sp_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if par_pat.search(n): pars.append(d/1e3)
        elif red_pat.search(n): reds.append(d/1e3)
print(f"par: {len(pars)}, red: {len(reds)}")
# per config: 3 warmup + 1 measured = 4 calls. measured = last par+reduce.
print(f"\n{'b×totlen':>14} {'par(us)':>8} {'reduce(us)':>10} {'total(us)':>9} {'reduce%':>8}")
for ci,(b,t) in enumerate(configs):
    pu = pars[ci*4+3]; ru = reds[ci*4+3]
    print(f"  {b:2d}×{t:5d}  {pu:6.1f}  {ru:8.1f}  {pu+ru:7.1f}  {ru/(pu+ru)*100:6.1f}%")
