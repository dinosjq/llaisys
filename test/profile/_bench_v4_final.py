#!/usr/bin/env python3
"""LLAISYS v4 final benchmark: NH=12, NKVH=2, HD=128, within BATCH_MAX_TOKEN_NUM=8192.
Matches the methodology from docs/optimization/2026-08-08-flash-decoding-tflop.md.
FLOPs = batch * nh * 4 * totlen * hd.  Bandwidth = (K+V+Q+out) bytes / time."""
import torch, sys, os, ctypes, time, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

# DeepSeek-R1-Distill-Qwen-1.5B: nh=12, nkvh=2, hd=128
NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2  # bf16

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# Configs within BATCH_MAX_TOKEN_NUM = 8192
CONFIGS = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (4,2048),(8,1024),(16,512),(32,256)]
# Also run the larger configs from the old report for comparison
CONFIGS += [(4,256),(4,512),(4,1024),(8,256),(8,512),(8,2048),(16,256),(16,1024),(16,2048)]

TESTS = [(6,8,1),(2,16,1),(6,8,4),(3,16,8)]  # v3 default, small-batch best, CO=4, CO=8 combo

# Generate nsys profile script
script = os.path.join(os.path.dirname(__file__), '_run_v4_bench.py')
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
configs = {repr(CONFIGS)}
tests = {repr(TESTS)}
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

print(f"Configs: {len(CONFIGS)} × {len(TESTS)} tests = {len(CONFIGS)*len(TESTS)} combos")
print("Running nsys...")
r = subprocess.run(
    ["nsys", "profile", "-t", "cuda,nvtx", "-o", "/tmp/v4_bench_final", "--force-overwrite=true",
     "python3", script],
    check=False, timeout=600, capture_output=True, text=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:]); sys.exit(1)

print("Exporting...")
subprocess.run(["rm", "-f", "/tmp/v4_bench_final_t*.csv"], check=False)
subprocess.run(
    ["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv", "--force-export=true",
     "-o", "/tmp/v4_bench_final_t", "/tmp/v4_bench_final.nsys-rep"],
    check=True, timeout=120, capture_output=True)

# Parse
par_pat = re.compile(r"flash_decoding_v4_kernel")
red_pat = re.compile(r"flash_decoding_v4_reduce")
kernels = []
with open("/tmp/v4_bench_final_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        name, dur = row[ni], int(row[di])
        if par_pat.search(name) or red_pat.search(name):
            kernels.append((name, dur))

assert len(kernels) == len(CONFIGS) * len(TESTS) * 4, f"{len(kernels)} vs {len(CONFIGS)*len(TESTS)*4}"

results = {}
for ci, (batch, totlen) in enumerate(CONFIGS):
    for ti, (h, tk, co) in enumerate(TESTS):
        idx = ci * len(TESTS) + ti
        base = idx * 4
        pu = kernels[base + 2][1] / 1e3
        ru = kernels[base + 3][1] / 1e3
        results[(batch, totlen, h, tk, co)] = (pu, ru, pu + ru)

# FLOPs and bandwidth
def calc_metrics(batch, totlen, total_us):
    flops = batch * NH * 4 * totlen * HD
    tflops = flops / (total_us * 1e-6) / 1e12 if total_us > 0 else 0
    # K cache + V cache + Q + output: all bf16 = 2 bytes/element
    kv_bytes = batch * totlen * NKVH * HD * 2 * 2  # K+V, 2 bytes per element
    q_bytes = batch * NH * HD * 2
    out_bytes = batch * NH * HD * 2
    total_bytes = kv_bytes + q_bytes + out_bytes
    gbs = total_bytes / (total_us * 1e-6) / 1e9 if total_us > 0 else 0
    return flops, tflops, total_bytes, gbs

# Print table
print(f"\n{'batch × totlen':>16} {'LLAISYS(us)':>12} {'TFLOPS':>8} {'GB/s':>8} {'best(H,TK,CO)':>14}")
print("-" * 70)
for batch, totlen in CONFIGS:
    best = 1e9; best_cfg = None
    for h, tk, co in TESTS:
        k = (batch, totlen, h, tk, co)
        if k in results and results[k][2] < best:
            best = results[k][2]; best_cfg = (h, tk, co)
    if best_cfg:
        _, tflops, _, gbs = calc_metrics(batch, totlen, best)
        h, tk, co = best_cfg
        print(f"  {batch:2d} × {totlen:4d}      {best:8.1f}     {tflops:5.2f}   {gbs:6.1f}     H={h} TK={tk} CO={co}")

# Comparison table with FlashInfer (from 2026-08-08 report, NH=12)
fi_data = {
    (1,256): 48.3, (1,512): 55.0, (1,1024): 48.8, (1,2048): 53.9,
    (4,256): 49.6, (4,512): 53.6, (4,1024): 58.0, (4,2048): 68.1,
    (8,256): 52.5, (8,512): 56.2, (8,1024): 68.4, (8,2048): 90.3,
    (16,256): 60.5, (16,512): 72.8, (16,1024): 90.1, (16,2048): 144.3,
}

print(f"\n{'batch × totlen':>16} {'LLAISYS(us)':>12} {'FlashInfer(us)':>14} {'LL/FI':>7} {'LL TFLOPS':>9} {'FI TFLOPS':>9}")
print("-" * 80)
ll_sum = 0; fi_sum = 0; n = 0
for batch, totlen in CONFIGS:
    best = 1e9
    for h, tk, co in TESTS:
        k = (batch, totlen, h, tk, co)
        if k in results and results[k][2] < best:
            best = results[k][2]
    fi = fi_data.get((batch, totlen))
    if fi:
        flops, ll_tf, _, _ = calc_metrics(batch, totlen, best)
        fi_tf = flops / (fi * 1e-6) / 1e12
        print(f"  {batch:2d} × {totlen:4d}      {best:8.1f}        {fi:8.1f}        {best/fi:.2f}x   {ll_tf:5.2f}     {fi_tf:5.2f}")
        ll_sum += best; fi_sum += fi; n += 1
if n > 0:
    print(f"  {'mean':>16}      {ll_sum/n:8.1f}        {fi_sum/n:8.1f}        {ll_sum/fi_sum:.2f}x")
