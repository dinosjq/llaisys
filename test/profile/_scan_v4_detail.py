#!/usr/bin/env python3
"""v4 coalesce 全细节扫描：每 config 展示全部 48 参数组合的 par/red/total 时间."""
import torch, sys, os, ctypes, csv, subprocess, re
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

CONFIGS = [(b, t) for b in [1, 4, 8, 16, 32] for t in [256, 512, 1024, 2048, 4096]]
HEADS_LIST = [1, 2, 3, 6]
TILE_K_LIST = [8, 16]
COALESCE_LIST = [1, 2, 3, 4, 6, 8]

combos = []
for batch, totlen in CONFIGS:
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                combos.append((batch, totlen, heads, tk, co))
print(f"Total combos: {len(combos)}")

# ── 生成 profile 脚本 ───────────────────────────────────────────
script = os.path.join(os.path.dirname(__file__), '_run_v4_detail.py')
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
combos = {repr(combos)}
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
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    label=f"B{{batch}}_T{{totlen}}_H{{heads}}_K{{tk}}_C{{co}}"
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

print(f"Profile script: {script}")
print("Running nsys...")
r = subprocess.run(
    ["nsys", "profile", "-t", "cuda,nvtx", "-o", "/tmp/v4_detail", "--force-overwrite=true",
     "python3", script],
    check=False, timeout=600, capture_output=True, text=True)
if r.returncode != 0:
    print("STDERR:", r.stderr[-2000:]); sys.exit(1)

print("Exporting...")
subprocess.run(["rm", "-f", "/tmp/v4_detail.sqlite", "/tmp/v4_detail_t*.csv"], check=False)
subprocess.run(
    ["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv", "--force-export=true",
     "-o", "/tmp/v4_detail_t", "/tmp/v4_detail.nsys-rep"],
    check=True, timeout=120, capture_output=True)

# ── 解析 ─────────────────────────────────────────────────────────
par_pat = re.compile(r"flash_decoding_v4_kernel")
red_pat = re.compile(r"flash_decoding_v4_reduce")
kernels = []
with open("/tmp/v4_detail_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        name, dur = row[ni], int(row[di])
        if par_pat.search(name) or red_pat.search(name):
            kernels.append((name, dur))

assert len(kernels) == len(combos) * 4, f"Expected {len(combos)*4}, got {len(kernels)}"

results = {}
for idx, (batch, totlen, heads, tk, co) in enumerate(combos):
    base = idx * 4
    par_us = kernels[base + 2][1] / 1e3
    red_us = kernels[base + 3][1] / 1e3
    results[(batch, totlen, heads, tk, co)] = (par_us, red_us, par_us + red_us)

# ── 详细输出 ─────────────────────────────────────────────────────
for batch, totlen in CONFIGS:
    bn = totlen // TN
    tokens = batch * bn * TN
    flops_per_token = 2 * NH * HD  # Q*K + A*V
    total_flops = batch * flops_per_token * totlen

    print(f"\n{'='*120}")
    print(f"batch={batch:3d}  totlen={totlen:6d}  blocks/seq={bn:4d}  tot_blocks={batch*bn:6d}  "
          f"KV_tokens={tokens:7d}  TFLOPs={total_flops/1e9:.3f}")
    print(f"{'='*120}")

    # 表1: 每个 HEADS 下 COALESCE 的影响 (TK=8)
    print(f"\n  ── TK=8: COALESCE vs HEADS (par+red us) ──")
    header = f"  {'HEADS':>6}"
    for co in COALESCE_LIST:
        header += f"  CO={co:>6}"
    header += f"  best_CO  vs_H6C1"
    print(header)
    for heads in HEADS_LIST:
        line = f"  {heads:6d}"
        best = 1e9; best_co = 0
        ref = results.get((batch, totlen, 6, 8, 1), (0,0,0))[2]
        for co in COALESCE_LIST:
            k = (batch, totlen, heads, 8, co)
            if k in results:
                t = results[k][2]
                line += f" {t:7.1f}"
                if t < best: best = t; best_co = co
            else:
                line += f" {'N/A':>7}"
        if ref > 0 and best < 1e8:
            line += f"  CO={best_co}  {best/ref:.2f}x"
        print(line)

    # 表2: 每个 HEADS 下 COALESCE 的影响 (TK=16)
    print(f"\n  ── TK=16: COALESCE vs HEADS (par+red us) ──")
    header = f"  {'HEADS':>6}"
    for co in COALESCE_LIST:
        header += f"  CO={co:>6}"
    header += f"  best_CO  vs_H6C1"
    print(header)
    for heads in HEADS_LIST:
        line = f"  {heads:6d}"
        best = 1e9; best_co = 0
        ref = results.get((batch, totlen, 6, 8, 1), (0,0,0))[2]
        for co in COALESCE_LIST:
            k = (batch, totlen, heads, 16, co)
            if k in results:
                t = results[k][2]
                line += f" {t:7.1f}"
                if t < best: best = t; best_co = co
            else:
                line += f" {'N/A':>7}"
        if ref > 0 and best < 1e8:
            line += f"  CO={best_co}  {best/ref:.2f}x"
        print(line)

    # 表3: 全部 48 组合排名 (top 10)
    print(f"\n  ── Top 10 参数组合 (par+red us) ──")
    all_combos = []
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                k = (batch, totlen, heads, tk, co)
                if k in results:
                    all_combos.append((heads, tk, co) + results[k])
    all_combos.sort(key=lambda x: x[5])
    print(f"  {'rank':>4} {'H':>4} {'TK':>4} {'CO':>4} {'par_us':>9} {'red_us':>9} {'total':>9} {'TFLOPs':>8} {'%best':>7}")
    if all_combos:
        best_t = all_combos[0][5]
        for i, (h, tk, co, pu, ru, tu) in enumerate(all_combos[:10]):
            tflops = total_flops / (tu * 1e-6) / 1e12 if tu > 0 else 0
            pct = tu / best_t * 100 if best_t > 0 else 0
            print(f"  {i+1:4d} {h:4d} {tk:4d} {co:4d} {pu:9.1f} {ru:9.1f} {tu:9.1f} {tflops:8.2f} {pct:7.1f}%")

print("\ndone.")
