#!/usr/bin/env python3
"""v6 vs v3 vs v5 benchmark."""
import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,2048),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),(16,256),(16,512),(32,256)]

# v3: (H,TK,use_prodcon=0). v5/v6: (H,TK,CO)
v3_tests = [(6,8,0),(2,16,0)]
vv_tests = [(6,8,1),(6,8,4),(3,16,4),(2,16,1),(3,16,8)]

script = os.path.join(os.path.dirname(__file__), '_run_v6.py')
with open(script,'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
for fn in ['llaisysFlashDecodingV3','llaisysFlashDecodingV5','llaisysFlashDecodingV6']:
    getattr(lib,fn).argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; v3_tests = {repr(v3_tests)}; vv_tests = {repr(vv_tests)}
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
        label=f"V3_B{{batch}}T{{totlen}}_H{{h}}K{{tk}}"
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
            label=f"{{fn[-2:]}}_B{{batch}}T{{totlen}}_H{{h}}K{{tk}}C{{co}}"
            torch.cuda.nvtx.range_push(label)
            getattr(lib,fn)(*args); torch.cuda.synchronize()
            torch.cuda.nvtx.range_pop()
print("done")
""")

print("Running nsys...")
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v6_bench","--force-overwrite=true","python3",script],
               check=False, timeout=600, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v6_bench_t*.csv"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v6_bench_t","/tmp/v6_bench.nsys-rep"], check=True, timeout=120, capture_output=True)

v3_par = re.compile(r"flash_decoding_v3_kernel"); v3_red = re.compile(r"flash_decoding_v3_reduce")
v5_par = re.compile(r"flash_decoding_v5_kernel"); v5_red = re.compile(r"flash_decoding_v5_reduce")
v6_par = re.compile(r"flash_decoding_v6_kernel"); v6_red = re.compile(r"flash_decoding_v6_reduce")
kernels = []
with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if any(p.search(n) for p in [v3_par,v3_red,v5_par,v5_red,v6_par,v6_red]):
            kernels.append((d, n))

per = 2*4 + 5*4 + 5*4  # 2 v3 + 5 v5 + 5 v6
assert len(kernels) == len(configs) * per, f"{len(kernels)} vs {len(configs)*per}"

def get_tot(bo): return (kernels[bo+2][0] + kernels[bo+3][0])/1e3

print(f"\n{'b×totlen':>14} {'v3_H6T8':>8} {'v5_best':>8} {'v6_best':>8} {'v6_cfg':>14} {'v6/v3':>7}")
for ci, (b,t) in enumerate(configs):
    base = ci * per
    v3h6 = get_tot(base)
    base5 = base + 8
    base6 = base5 + 5*4
    v5_best = min(get_tot(base5 + ti*4) for ti in range(5))
    v6_vals = [(get_tot(base6 + ti*4), vv_tests[ti]) for ti in range(5)]
    v6_best, v6_cfg = min(v6_vals)
    print(f"  {b:2d}×{t:5d}  {v3h6:6.1f}  {v5_best:6.1f}  {v6_best:6.1f}  H={v6_cfg[0]}T{v6_cfg[1]}C{v6_cfg[2]}  {v6_best/v3h6:.2f}x")

# CO detail for v6
print(f"\n{'b×totlen':>14} | v6 (H=6,TK=8): C1..C4 | v6 (H=3,TK=16): C1,C4,C8")
for ci, (b,t) in enumerate(configs):
    base = ci * per
    base6 = base + 8 + 5*4
    # vv_tests = [(6,8,1),(6,8,4),(3,16,4),(2,16,1),(3,16,8)]
    v6_68c1 = get_tot(base6); v6_68c4 = get_tot(base6+4)
    v6_316c4 = get_tot(base6+8); v6_216c1 = get_tot(base6+12); v6_316c8 = get_tot(base6+16)
    print(f"  {b:2d}×{t:5d} | {v6_68c1:6.1f} {v6_68c4:6.1f}      | {v6_316c4:6.1f} {v6_216c1:6.1f} {v6_316c8:6.1f}")
