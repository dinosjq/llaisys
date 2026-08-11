#!/usr/bin/env python3
"""v4 coalesce 全网格扫描: HEADS × TILE_K × COALESCE（含 CO=3,6）.
每个 batch 使用不重叠 block IDs，模拟真实 KV cache 访问模式."""
import torch, sys, os, ctypes, csv, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV4.argtypes = [
    c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# ── 配置 ─────────────────────────────────────────────────────────
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

# ── 生成 nsys profile 脚本 ───────────────────────────────────────
profile_script = os.path.join(os.path.dirname(__file__), '_run_v4_full.py')
with open(profile_script, 'w') as f:
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
    bn = totlen // TN; mb = bn + 1; tb = batch * bn
    # 不重叠 block IDs: batch i 使用 blocks [i*bn, (i+1)*bn)
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b+1)*bn
        tbids[b, 1:] = torch.arange(b*bn, (b+1)*bn, device=DEV)
    qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    kt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    vt = torch.randn((tb, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    cut = torch.arange(batch+1, dtype=torch.int64, device=DEV)
    tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
    acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
    asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    s = 1.0/(HD**0.5)
    args = (av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
            qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
            TN,batch,mb,tb,1,BV2,s,NH,HD,HD,NKVH,heads,tk,co)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    label = f"b{{batch}}t{{totlen}}_H{{heads}}_TK{{tk}}_CO{{co}}"
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

print(f"Profile script: {profile_script}")
print("Running nsys profile...")
result = subprocess.run(
    ["nsys", "profile", "-t", "cuda,nvtx", "-o", "/tmp/v4_full", "--force-overwrite=true",
     "python3", profile_script],
    check=False, timeout=600, capture_output=True, text=True)
if result.returncode != 0:
    print("STDERR:", result.stderr[-3000:])
    sys.exit(1)

print("Exporting trace...")
subprocess.run(["rm", "-f", "/tmp/v4_full.sqlite", "/tmp/v4_full_t*.csv"], check=False)
subprocess.run(
    ["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv",
     "-o", "/tmp/v4_full_t", "/tmp/v4_full.nsys-rep"],
    check=True, timeout=120, capture_output=True)

# ── 解析（位置映射法）────────────────────────────────────────────
import re
par_pat = re.compile(r"flash_decoding_v4_kernel")
red_pat = re.compile(r"flash_decoding_v4_reduce")

kernels = []
with open("/tmp/v4_full_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        name, dur = row[ni], int(row[di])
        if par_pat.search(name) or red_pat.search(name):
            kernels.append((name, dur))

print(f"\nTotal v4 kernel entries: {len(kernels)} (expected {len(combos)*4})")
assert len(kernels) == len(combos) * 4, f"Mismatch: {len(kernels)} vs {len(combos)*4}"

results = {}
for idx, (batch, totlen, heads, tk, co) in enumerate(combos):
    base = idx * 4
    par_us = kernels[base + 2][1] / 1e3
    red_us = kernels[base + 3][1] / 1e3
    results[(batch, totlen, heads, tk, co)] = (par_us, red_us, par_us + red_us)

# ── 最优参数表 ───────────────────────────────────────────────────
print(f"\n{'batch':>5} {'totlen':>6} | {'best_h':>6} {'best_tk':>7} {'best_co':>7} {'par_us':>8} {'red_us':>8} {'total':>8} | {'v3(H6T8)':>9} {'speedup':>8}")
print("-" * 120)

for batch, totlen in CONFIGS:
    best_t = 1e9; best = None
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                key = (batch, totlen, heads, tk, co)
                if key in results:
                    _, _, total = results[key]
                    if total < best_t:
                        best_t = total
                        best = (heads, tk, co, results[key])
    if best:
        h, tk, co, (pu, ru, tu) = best
        v3_key = (batch, totlen, 6, 8, 1)
        v3_total = results.get(v3_key, (0,0,0))[2]
        sp = f"{v3_total/tu:.2f}x" if v3_total > 0 else "N/A"
        print(f"{batch:5d} {totlen:6d} | {h:6d} {tk:7d} {co:7d} {pu:8.1f} {ru:8.1f} {tu:8.1f} | {v3_total:9.1f} {sp:>8}")

# ── COALESCE 有效性矩阵 (固定 H=6, TK=8) ───────────────────────
print("\n=== COALESCE 有效性 (H=6, TK=8, par+red us) ===")
header = f"{'b\\totlen':>9}"
for t in sorted(set(c[1] for c in CONFIGS)):
    header += f" {t:>8}"
print(header)
for b in sorted(set(c[0] for c in CONFIGS)):
    line = f"{'b='+str(b):>9}"
    for t in sorted(set(c[1] for c in CONFIGS)):
        best_co_t = 1; best_us = 1e9
        for co in COALESCE_LIST:
            key = (b, t, 6, 8, co)
            if key in results:
                _, _, total = results[key]
                if total < best_us:
                    best_us = total; best_co_t = co
        v3_us = results.get((b, t, 6, 8, 1), (0,0,0))[2]
        sp = f"{(v3_us/best_us - 1)*100:+.0f}%" if v3_us > 0 else ""
        line += f" {best_co_t}({best_us:5.0f})" if best_us < 1e8 else "     N/A"
    print(line)

# ── 按 batch 的最优参数组 ────────────────────────────────────────
print("\n=== 按 batch 分组最优参数 ===")
for b in sorted(set(c[0] for c in CONFIGS)):
    print(f"\nbatch={b}:")
    print(f"  {'totlen':>6} {'h':>3} {'tk':>3} {'co':>3} {'par':>8} {'red':>8} {'total':>8}")
    for batch, totlen in [(bt, tt) for (bt, tt) in CONFIGS if bt == b]:
        best_t = 1e9; best = None
        for heads in HEADS_LIST:
            for tk in TILE_K_LIST:
                for co in COALESCE_LIST:
                    key = (batch, totlen, heads, tk, co)
                    if key in results:
                        _, _, total = results[key]
                        if total < best_t:
                            best_t = total
                            best = (heads, tk, co, results[key])
        if best:
            h, tk, co, (pu, ru, tu) = best
            print(f"  {totlen:6d} {h:3d} {tk:3d} {co:3d} {pu:8.1f} {ru:8.1f} {tu:8.1f}")
