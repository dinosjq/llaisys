#!/usr/bin/env python3
"""v6 全参数扫描: HEADS × TILE_K × COALESCE × NUM_STAGES × UNROLL.
分片到 4 个 nsys shard, 覆盖关键 config. 位置映射解析."""
import torch, sys, os, ctypes, csv, subprocess, re, math, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0"); BV2 = 2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV6.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*5

HEADS = [1, 2, 3, 6]
TKS = [8, 16]
COS = [1, 2, 4]
NSS = [2, 3]
URS = [0, 1]

KEY = [(1,256),(1,1024),(1,2048),(1,4096),(4,512),(4,2048),(8,1024),(16,512),(32,256)]

# Global combo order (config-major, then h,tk,co,ns,ur)
combos = []
for b, t in KEY:
    for h in HEADS:
        for tk in TKS:
            for co in COS:
                for ns in NSS:
                    for ur in URS:
                        combos.append((b, t, h, tk, co, ns, ur))

n_shards = 4
shard_size = math.ceil(len(combos) / n_shards)
shards = [combos[i:i+shard_size] for i in range(0, len(combos), shard_size)]

# Generate shard scripts
for si, shard in enumerate(shards):
    script = os.path.join(os.path.dirname(__file__), f'_run_v6opt_s{si}.py')
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
combos = {repr(shard)}
for (batch, totlen, heads, tk, co, ns, ur) in combos:
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
          TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,heads,tk,co,ns,ur)
    for _ in range(3): lib.llaisysFlashDecodingV6(*args)   # warmup (含 JIT 编译)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(f"B{{batch}}T{{totlen}}_H{{heads}}K{{tk}}C{{co}}S{{ns}}U{{ur}}")
    lib.llaisysFlashDecodingV6(*args)                       # measured
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

# Run shards
all_par = []
for si in range(n_shards):
    print(f"Shard {si}/{n_shards} ({len(shards[si])} combos)...")
    r = subprocess.run(["nsys","profile","-t","cuda,nvtx","-o",f"/tmp/v6opt_s{si}","--force-overwrite=true",
                        "python3",os.path.join(os.path.dirname(__file__),f'_run_v6opt_s{si}.py')],
                       check=False, timeout=900, capture_output=True, text=True)
    if r.returncode != 0: print(f"  FAIL: {r.stderr[-300:]}"); sys.exit(1)
    subprocess.run(["rm","-f",f"/tmp/v6opt_s{si}_t*.csv"], check=False)
    subprocess.run(["nsys","stats","--report=cuda_gpu_trace","--format=csv","--force-export=true",
                    "-o",f"/tmp/v6opt_s{si}_t",f"/tmp/v6opt_s{si}.nsys-rep"], check=True, timeout=120, capture_output=True)
    par_durs = []
    with open(f"/tmp/v6opt_s{si}_t_cuda_gpu_trace.csv") as f:
        r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
        for row in r:
            n = row[ni]; d = int(row[di])
            if 'flash_decoding_v6_kernel' in n:
                par_durs.append(d/1e3)
    # 4 par per combo (3 warmup + 1 measured) -> last (index 3 mod 4) = measured
    for i in range(3, len(par_durs), 4):
        all_par.append(par_durs[i])

print(f"Measured par entries: {len(all_par)} (expected {len(combos)})")
assert len(all_par) == len(combos)

results = {}
for ci, combo in enumerate(combos):
    results[combo] = all_par[ci]

with open('/tmp/v6opt_results.json', 'w') as f:
    json.dump({str(k): v for k, v in results.items()}, f)

print(f"\n{'='*100}")
print("v6 FULL OPT PARAM SWEEP (par kernel us, 9 configs x 96 combos)")
print(f"{'='*100}")
for b, t in KEY:
    print(f"\n── batch={b} totlen={t} ──")
    cfg_combos = [c for c in combos if c[0]==b and c[1]==t]
    ranked = sorted(cfg_combos, key=lambda c: results[c])
    print(f"  Top 5:")
    for c in ranked[:5]:
        h, tk, co, ns, ur = c[2], c[3], c[4], c[5], c[6]
        print(f"    H{h} TK{tk} CO{co} NS{ns} UR{ur}: {results[c]:6.1f}us")
