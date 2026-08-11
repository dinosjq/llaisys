import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device("cuda:0"); BV2=2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),(4,2048),(8,1024),(16,512),(32,256),
           (2,256),(2,2048),(4,1024),(8,512),(16,256)]

# v3 uses (H,TK,use_prodcon) where 0=all-participate (fast), NOT coalesce!
v3_tests = [(6,8,0),(2,16,0)]
# v4/v5 use (H,TK,COALESCE)
v4v5_tests = [(6,8,1),(6,8,2),(6,8,4),(6,8,8),(2,16,1),(3,16,8)]

script = os.path.join(os.path.dirname(__file__), '_run_v5_full.py')
with open(script,'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = lib.llaisysFlashDecodingV4.argtypes = lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; v3_tests = {repr(v3_tests)}; v4v5_tests = {repr(v4v5_tests)}
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
              TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,0)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        label=f"V3_B{{batch}}T{{totlen}}_H{{h}}K{{tk}}"
        torch.cuda.nvtx.range_push(label)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
    for (h,tk,co) in v4v5_tests:
        for fn in ['llaisysFlashDecodingV4','llaisysFlashDecodingV5']:
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
subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v5_full","--force-overwrite=true","python3",script],
               check=False, timeout=600, capture_output=True, text=True)
subprocess.run(["rm","-f","/tmp/v5_full_t*.csv"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v5_full_t","/tmp/v5_full.nsys-rep"], check=True, timeout=120, capture_output=True)

# Parse
v3_par = re.compile(r"flash_decoding_v3_kernel"); v3_red = re.compile(r"flash_decoding_v3_reduce")
v4_par = re.compile(r"flash_decoding_v4_kernel"); v4_red = re.compile(r"flash_decoding_v4_reduce")
v5_par = re.compile(r"flash_decoding_v5_kernel"); v5_red = re.compile(r"flash_decoding_v5_reduce")
kernels = []
with open("/tmp/v5_full_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        if any(p.search(n) for p in [v3_par,v3_red,v4_par,v4_red,v5_par,v5_red]):
            kernels.append((d, n))

# Per-config entries: 2 v3 combos × 4 + 6 v4 combos × 4 + 6 v5 combos × 4 = 8+24+24 = 56
per = 2*4 + 6*4 + 6*4
assert len(kernels) == len(configs) * per, f"{len(kernels)} vs {len(configs)*per}"

def get_total(entries, base):
    return (entries[base+2][0] + entries[base+3][0]) / 1e3

print(f"\n{'batch×totlen':>14} {'v3(H6T8)':>9} {'v3(H2T16)':>9} {'v4(H6T8C1)':>10} {'v4(H6T8C4)':>10} {'v5(H6T8C1)':>10} {'v5(H6T8C4)':>10} {'v5_best':>9} {'best(H,TK,CO)':>14}")
print("-"*120)
for ci, (batch, totlen) in enumerate(configs):
    base = ci * per
    # v3: 2 combos × 4 entries each. get_total reads entries[bo+2]+[bo+3]
    v3h6 = get_total(kernels, base)        # combo 0 (H6T8)
    v3h2 = get_total(kernels, base + 4)    # combo 1 (H2T16) — skip 4 entries
    base4 = base + 8                        # v4 starts after 2 v3 combos (8 entries)
    v4c1 = get_total(kernels, base4)        # v4 combo 0 (H6T8C1)
    v4c4 = get_total(kernels, base4 + 3*4)  # v4 combo 3 (H6T8C4)
    base5 = base4 + 6*4                     # v5 starts after 6 v4 combos (24 entries)
    v5c1 = get_total(kernels, base5)        # v5 combo 0 (H6T8C1)
    v5c4 = get_total(kernels, base5 + 3*4)  # v5 combo 3 (H6T8C4)

    # Find v5 best across all 6 combos
    v5_best = 1e9; v5_best_cfg = None
    for ti, (h,tk,co) in enumerate(v4v5_tests):
        t = get_total(kernels, base5 + ti*4)
        if t < v5_best: v5_best = t; v5_best_cfg = (h,tk,co)
    print(f"  {batch:2d} × {totlen:5d}  {v3h6:8.1f} {v3h2:8.1f} {v4c1:9.1f} {v4c4:9.1f} {v5c1:9.1f} {v5c4:9.1f} {v5_best:8.1f}  H={v5_best_cfg[0]} TK={v5_best_cfg[1]} CO={v5_best_cfg[2]}")
