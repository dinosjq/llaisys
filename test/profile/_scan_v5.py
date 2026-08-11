#!/usr/bin/env python3
"""v5 full parameter sweep: HEADS×TILE_K×COALESCE, dispatched to parallel agents."""
import torch, sys, os, ctypes, csv, subprocess, re, json, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0"); BV2 = 2

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# Full parameter space
HEADS_LIST = [1, 2, 3, 6]
TILE_K_LIST = [8, 16]
COALESCE_LIST = [1, 2, 3, 4, 6, 8]

# System-valid configs
def valid(b, t):
    return b * t <= 8192 and b * ((t + 63) // 64) <= 128 and (t <= 2048 or (b == 1 and t == 4096))
configs = []
for b in [1,2,4,8,16,32]:
    for t in [256,512,1024,2048]:
        if valid(b, t): configs.append((b, t))
configs.append((1, 4096))

# Build all combos
combos = []
for b, t in configs:
    for h in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                combos.append((b, t, h, tk, co))

print(f"Total: {len(configs)} configs × 48 params = {len(combos)} combos")

# Split into 4 shards for parallel nsys runs
n_shards = 4
shard_size = math.ceil(len(combos) / n_shards)
shards = [combos[i:i+shard_size] for i in range(0, len(combos), shard_size)]

for si, shard in enumerate(shards):
    script = os.path.join(os.path.dirname(__file__), f'_run_v5_s{si}.py')
    with open(script, 'w') as f:
        f.write(f"""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..')}')
sys.path.insert(0, r'{os.path.join(os.path.dirname(__file__), '..', '..', 'python')}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV5.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN=12,2,128,64; DEV=torch.device('cuda:0'); BV2=2
combos = {repr(shard)}
for (batch, totlen, heads, tk, co) in combos:
    bn=totlen//TN; mb=bn+1; tb=batch*bn
    tbids=torch.zeros((batch,mb),dtype=torch.int64,device=DEV)
    for b in range(batch):
        tbids[b,0]=(b+1)*bn; tbids[b,1:]=torch.arange(b*bn,(b+1)*bn,device=DEV)
    qt=torch.randn((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    kt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    vt=torch.randn((tb,TN,NKVH,HD),dtype=torch.bfloat16,device=DEV)
    av=torch.empty((batch,NH,HD),dtype=torch.bfloat16,device=DEV)
    cut=torch.arange(batch+1,dtype=torch.int64,device=DEV)
    tot=torch.full((batch,),totlen,dtype=torch.int64,device=DEV)
    acc=torch.zeros((tb,NH,HD),dtype=torch.float32,device=DEV)
    asum=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    amax=torch.zeros((tb,NH,1),dtype=torch.float32,device=DEV)
    s=1.0/(HD**0.5)
    args=(av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
          qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
          TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,heads,tk,co)
    lib.llaisysFlashDecodingV5(*args); torch.cuda.synchronize()
    label=f"B{{batch}}T{{totlen}}_H{{heads}}K{{tk}}C{{co}}"
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV5(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")
    print(f"Shard {si}: {len(shard)} combos → {script}")

# Run all shards with nsys (sequentially since nsys captures whole GPU)
all_kernels = []
for si in range(n_shards):
    print(f"\nRunning shard {si}/{n_shards}...")
    r = subprocess.run(
        ["nsys", "profile", "-t", "cuda,nvtx", "-o", f"/tmp/v5_s{si}", "--force-overwrite=true",
         "python3", os.path.join(os.path.dirname(__file__), f'_run_v5_s{si}.py')],
        check=False, timeout=900, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-500:]}")
        continue
    subprocess.run(["rm", "-f", f"/tmp/v5_s{si}_t*.csv"], check=False)
    subprocess.run(
        ["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv", "--force-export=true",
         "-o", f"/tmp/v5_s{si}_t", f"/tmp/v5_s{si}.nsys-rep"],
        check=True, timeout=120, capture_output=True)

    par_pat = re.compile(r"flash_decoding_v5_kernel")
    red_pat = re.compile(r"flash_decoding_v5_reduce")
    with open(f"/tmp/v5_s{si}_t_cuda_gpu_trace.csv") as f:
        r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
        for row in r:
            n = row[ni]; d = int(row[di])
            if par_pat.search(n) or red_pat.search(n):
                all_kernels.append((d, n))
    print(f"  Kernels collected: {len(all_kernels)} total")

# Parse all results
assert len(all_kernels) == len(combos) * 4, f"Expected {len(combos)*4}, got {len(all_kernels)}"

results = {}
for idx, (b, t, h, tk, co) in enumerate(combos):
    base = idx * 4
    pu = all_kernels[base+2][0] / 1e3
    ru = all_kernels[base+3][0] / 1e3
    results[(b, t, h, tk, co)] = (pu, ru, pu+ru)

# Save results
import json
with open('/tmp/v5_results.json', 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f)

# Print report
print(f"\n{'='*100}")
print(f"v5 FULL PARAMETER SWEEP: {len(configs)} configs × 48 params = {len(combos)} combos")
print(f"{'='*100}")

for batch, totlen in configs:
    bn = totlen // TN
    tokens = batch * bn * TN
    print(f"\n── batch={batch:2d} totlen={totlen:5d}  blocks/seq={bn:3d}  tot_blocks={batch*bn:4d}  KV_tokens={tokens:6d} ──")

    # Best 10 overall
    all_vals = []
    for h in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                k = (batch, totlen, h, tk, co)
                if k in results:
                    all_vals.append((results[k][2], h, tk, co))
    all_vals.sort()
    print(f"  Top 5:")
    for i, (t, h, tk, co) in enumerate(all_vals[:5]):
        flops = batch * NH * 4 * totlen * HD
        tflops = flops / (t * 1e-6) / 1e12
        print(f"    {i+1}. H={h} TK={tk} CO={co:2d}  {t:8.1f}us  {tflops:5.2f} TFLOPS")

    # COALESCE isolated: for each (H,TK), show all CO values
    for h in [6, 3]:
        for tk in [8, 16]:
            print(f"  H={h} TK={tk}: ", end="")
            for co in COALESCE_LIST:
                k = (batch, totlen, h, tk, co)
                if k in results:
                    print(f"CO{co}={results[k][2]:.1f} ", end="")
            print()
