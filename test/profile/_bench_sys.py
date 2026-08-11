#!/usr/bin/env python3
"""LLAISYS v4 final — system config: TN=64, NH=12, NKVH=2, HD=128, MAX_TOKEN=2048, BATCH_MAX=8192."""
import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
BV2 = 2; DEV = torch.device("cuda:0")
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# Valid configs: batch*totlen<=8192, totlen<=2048, batch*ceil(totlen/64)<=128
def valid(b,t):
    return b*t <= 8192 and t <= 2048 and b*((t+63)//64) <= 128

configs = []
for b in [1,2,4,8,16,32]:
    for t in [256,512,1024,2048]:
        if valid(b,t): configs.append((b,t))
# Also add 1x4096 (valid: 1*4096<=8192, 64 blocks<=128)
for b,t in [(1,4096)]:
    if valid(b,t): configs.append((b,t))

TESTS = [(6,8,1),(2,16,1),(3,16,8)]

script = os.path.join(os.path.dirname(__file__), '_run_sys.py')
with open(script, 'w') as f:
    f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
configs = {repr(configs)}; tests = {repr(TESTS)}
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
    for (h,tk,co) in tests:
        av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
        acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
        asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
        amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
        args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
              qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
              TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,h,tk,co)
        lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
        label=f"B{{batch}}T{{totlen}}_H{{h}}K{{tk}}C{{co}}"
        torch.cuda.nvtx.range_push(label)
        lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
print("done")
""")

print(f"Configs: {len(configs)} x {len(TESTS)} tests")
r = subprocess.run(["nsys","profile","-t","cuda,nvtx","-o","/tmp/v4_sys","--force-overwrite=true",
                    "python3",script], check=False, timeout=600, capture_output=True, text=True)
if r.returncode != 0: print("STDERR:",r.stderr[-2000:]); sys.exit(1)

subprocess.run(["rm","-f","/tmp/v4_sys_t*.csv"], check=False)
subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                "-o","/tmp/v4_sys_t","/tmp/v4_sys.nsys-rep"], check=True, timeout=120, capture_output=True)

par_pat = re.compile(r"flash_decoding_v4_kernel")
red_pat = re.compile(r"flash_decoding_v4_reduce")
kernels = []
with open("/tmp/v4_sys_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        name, dur = row[ni], int(row[di])
        if par_pat.search(name) or red_pat.search(name): kernels.append((name, dur))

assert len(kernels) == len(configs) * len(TESTS) * 4
results = {}
for ci, (batch, totlen) in enumerate(configs):
    for ti, (h, tk, co) in enumerate(TESTS):
        idx = ci * len(TESTS) + ti; base = idx * 4
        pu = kernels[base+2][1]/1e3; ru = kernels[base+3][1]/1e3
        results[(batch,totlen,h,tk,co)] = (pu, ru, pu+ru)

# Print table
print(f"\n{'batch×totlen':>14} {'v4_best(us)':>12} {'par':>8} {'red':>8} {'TFLOPS':>8} {'GB/s':>8} {'best(H,TK,CO)':>14}  {'blk/seq':>7} {'grid':>7}")
print("-"*110)
for batch, totlen in configs:
    best = 1e9; best_cfg = None
    for h,tk,co in TESTS:
        k = (batch,totlen,h,tk,co)
        if k in results and results[k][2] < best: best = results[k][2]; best_cfg = (h,tk,co,results[k])
    if best_cfg:
        h,tk,co,(pu,ru,tu) = best_cfg
        flops = batch * NH * 4 * totlen * HD
        tflops = flops / (tu*1e-6) / 1e12
        kvb = batch * totlen * NKVH * HD * 4
        qb = batch * NH * HD * 2; ob = batch * NH * HD * 2
        gbs = (kvb+qb+ob) / (tu*1e-6) / 1e9
        bn = totlen//TN
        grid = batch * ((bn + co - 1)//co) * (NH//h)
        print(f"  {batch:2d} × {totlen:5d}      {tu:8.1f}   {pu:6.1f}  {ru:6.1f}   {tflops:5.2f}   {gbs:6.1f}     H={h} TK={tk} CO={co}      {bn:4d}   {grid:6d}")

# Summary: v3 default vs best
print(f"\n{'batch×totlen':>14} {'v3(H6T8)(us)':>13} {'best(us)':>10} {'speedup':>8} {'best_cfg':>14}")
for batch, totlen in configs:
    v3k = (batch,totlen,6,8,1); bk = None; bv = 1e9
    for h,tk,co in TESTS:
        k = (batch,totlen,h,tk,co)
        if k in results and results[k][2] < bv: bv = results[k][2]; bk = (h,tk,co)
    if v3k in results and bk:
        v3t = results[v3k][2]; h,tk,co = bk
        print(f"  {batch:2d} × {totlen:5d}      {v3t:8.1f}     {bv:8.1f}    {v3t/bv:.2f}x     H={h} TK={tk} CO={co}")
