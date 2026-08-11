#!/usr/bin/env python3
"""v4 coalesce 全网格参数扫描: HEADS × TILE_K × COALESCE.
通过 nsys 精确测 parallel + reduce kernel GPU 时间。
正确性: v4 COALESCE=1 与 v3 对比."""
import torch, sys, os, ctypes, csv, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2  # bf16

lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)

# v3: last 3 ints = heads, tile_k, use_prodcon
lib.llaisysFlashDecodingV3.argtypes = [
    c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# v4: last 3 ints = heads, tile_k, coalesce
lib.llaisysFlashDecodingV4.argtypes = [
    c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

# ── Configs ─────────────────────────────────────────────────────
CONFIGS = [(b, t) for b in [1, 4, 8, 16, 32] for t in [256, 512, 1024, 2048, 4096]]
HEADS_LIST = [1, 2, 3, 6]
TILE_K_LIST = [8, 16]
COALESCE_LIST = [1, 2, 4, 8]

def build_block_ids(batch, totlen, token_num):
    """构建 block_ids: [batch, 1+block_num]，首列前缀和."""
    bn = totlen // token_num
    mb = bn + 1
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b + 1) * bn
        tbids[b, 1:] = torch.arange(bn, device=DEV)
    return tbids, bn

# ── 正确性验证: v4(CO=1) vs v3 ──────────────────────────────────
print("=== 正确性验证: v4(CO=1) vs v3 ===")
all_ok = True
for batch, totlen in [(1, 512), (4, 1024), (16, 2048)]:
    bn = totlen // TN; mb = bn + 1
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b + 1) * bn; tbids[b, 1:] = torch.arange(bn, device=DEV)
    qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    kt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    vt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    tb = batch * bn

    def run_v3():
        av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
        cut = torch.arange(batch + 1, dtype=torch.int64, device=DEV)
        tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
        acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
        asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        s = 1.0 / (HD ** 0.5)
        args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
                qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), tbids.data_ptr(),
                cut.data_ptr(), tot.data_ptr(),
                TN, batch, mb, tb, 1, BV2, s, NH, HD, HD, NKVH, 6, 8, 0)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        return av

    def run_v4(co):
        av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
        cut = torch.arange(batch + 1, dtype=torch.int64, device=DEV)
        tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
        acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
        asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
        s = 1.0 / (HD ** 0.5)
        args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
                qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), tbids.data_ptr(),
                cut.data_ptr(), tot.data_ptr(),
                TN, batch, mb, tb, 1, BV2, s, NH, HD, HD, NKVH, 6, 8, co)
        lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
        return av

    v3_ref = run_v3()
    for co in COALESCE_LIST:
        v4_out = run_v4(co)
        diff = (v3_ref.float() - v4_out.float()).abs().max().item()
        ok = diff < 0.5  # bf16 量化误差容忍
        status = "OK" if ok else f"FAIL(diff={diff:.4f})"
        print(f"  b={batch} t={totlen} co={co}: {status}")
        if not ok: all_ok = False

if not all_ok:
    print("\n❌ 正确性验证失败，请检查!")
    sys.exit(1)
print("✅ 正确性验证全部通过\n")

# ── nsys profile 脚本生成 ───────────────────────────────────────
profile_script = os.path.join(os.path.dirname(__file__), '_run_v4_grid.py')

# 构建所有参数组合的列表
combos = []
for batch, totlen in CONFIGS:
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                combos.append((batch, totlen, heads, tk, co))

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
    bn = totlen // TN; mb = bn + 1
    tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
    for b in range(batch):
        tbids[b, 0] = (b + 1) * bn; tbids[b, 1:] = torch.arange(bn, device=DEV)
    qt = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    kt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    vt = torch.randn((bn, TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
    tb = batch * bn
    av = torch.empty((batch, NH, HD), dtype=torch.bfloat16, device=DEV)
    cut = torch.arange(batch + 1, dtype=torch.int64, device=DEV)
    tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
    acc = torch.zeros((tb, NH, HD), dtype=torch.float32, device=DEV)
    asum = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    amax = torch.zeros((tb, NH, 1), dtype=torch.float32, device=DEV)
    s = 1.0 / (HD ** 0.5)
    args = (av.data_ptr(), acc.data_ptr(), asum.data_ptr(), amax.data_ptr(),
            qt.data_ptr(), kt.data_ptr(), vt.data_ptr(), tbids.data_ptr(),
            cut.data_ptr(), tot.data_ptr(),
            TN, batch, mb, tb, 1, BV2, s, NH, HD, HD, NKVH, heads, tk, co)
    label = f"b{{batch}}t{{totlen}}_H{{heads}}_TK{{tk}}_CO{{co}}"
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(label)
    lib.llaisysFlashDecodingV4(*args); torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()
print("done")
""")

print(f"Profile script: {profile_script}")
print(f"Total combinations: {len(combos)}")
print("Running nsys profile...")

# 跑 nsys
result = subprocess.run(
    ["nsys", "profile", "-t", "cuda,nvtx", "-o", "/tmp/v4_grid", "--force-overwrite=true",
     "python3", profile_script],
    check=False, timeout=600, capture_output=True, text=True)
if result.returncode != 0:
    print("STDERR:", result.stderr[-2000:])
    print("STDOUT:", result.stdout[-2000:])
    sys.exit(1)

print("Exporting trace...")
subprocess.run(["rm", "-f", "/tmp/v4_grid.sqlite", "/tmp/v4_grid_t*.csv"], check=False)
subprocess.run(
    ["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv",
     "-o", "/tmp/v4_grid_t", "/tmp/v4_grid.nsys-rep"],
    check=True, timeout=120, capture_output=True)

# ── 解析（位置映射法：按 combo 顺序匹配 kernel 对）────────────
print("\n=== v4 全网格扫描 (parallel + reduce kernel, us) ===\n")

import re
par_pat = re.compile(r"flash_decoding_v4_kernel")
red_pat = re.compile(r"flash_decoding_v4_reduce")
# 从 kernel 名提取模板参数 (HD, HEADS, TILE_K, COALESCE)
ktmpl_pat = re.compile(r"flash_decoding_v4_kernel.*?\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+)\)")

# 读取所有 v4 kernel + reduce 条目（保持顺序）
kernels = []
with open("/tmp/v4_grid_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        name, dur = row[ni], int(row[di])
        if par_pat.search(name) or red_pat.search(name):
            kernels.append((name, dur))

print(f"Total v4 kernel entries: {len(kernels)}")

# 每 combo 产生 4 个 kernel: warmup_par, warmup_red, meas_par, meas_red
# 对于 800 个 combo，共 3200 个 kernel entry
assert len(kernels) == len(combos) * 4, f"Expected {len(combos)*4} kernels, got {len(kernels)}"

# 提取 measured 部分: 每 4 个 entries = 1 combo，取后两个(meas_par, meas_red)
results = {}
for idx, (batch, totlen, heads, tk, co) in enumerate(combos):
    base = idx * 4
    meas_par_dur = kernels[base + 2][1] / 1e3  # measured parallel
    meas_red_dur = kernels[base + 3][1] / 1e3  # measured reduce
    results[(batch, totlen, heads, tk, co)] = (meas_par_dur, meas_red_dur, meas_par_dur + meas_red_dur)

# ── 输出表格 ─────────────────────────────────────────────────────
# 对每个 (batch, totlen)，找最优 (heads, tile_k, coalesce)
print("每个配置的最优参数 (按 total parallel+reduce us):")
print(f"{'batch':>5} {'totlen':>6} | {'best_h':>6} {'best_tk':>7} {'best_co':>7} {'par_us':>8} {'red_us':>8} {'total':>8} |", end="")
# 对比 v3(CO=1) baseline
print(f" {'v3_total':>8} {'speedup':>8}")
print("-" * 110)

for batch, totlen in CONFIGS:
    best = None
    best_total = 1e9
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                key = (batch, totlen, heads, tk, co)
                if key in results:
                    _, _, total = results[key]
                    if total < best_total:
                        best_total = total
                        best = (heads, tk, co, results[key])
    if best:
        h, tk, co, (pu, ru, tu) = best
        # v3 baseline = same config with co=1, H=6, TK=8
        v3_key = (batch, totlen, 6, 8, 1)
        v3_total = results.get(v3_key, (0, 0, 0))[2]
        sp = f"{v3_total/tu:.2f}x" if v3_total > 0 else "N/A"
        print(f"{batch:5d} {totlen:6d} | {h:6d} {tk:7d} {co:7d} {pu:8.1f} {ru:8.1f} {tu:8.1f} | {v3_total:8.1f} {sp:>8}")

# ── 详细矩阵: 固定 batch×totlen, 展示全部 (heads, coalesce) ─────
print("\n\n=== 详细矩阵 (TILE_K=8, 每个 batch×totlen 的 heads×coalesce) ===\n")
for batch, totlen in CONFIGS:
    print(f"\nbatch={batch}, totlen={totlen}:")
    header = f"{'heads\\co':>8}"
    for co in COALESCE_LIST:
        header += f" {'CO='+str(co):>10}"
    print(header)
    for heads in HEADS_LIST:
        line = f"{'H='+str(heads):>8}"
        for co in COALESCE_LIST:
            key = (batch, totlen, heads, 8, co)
            if key in results:
                _, _, total = results[key]
                line += f" {total:10.1f}"
            else:
                line += f" {'N/A':>10}"
        print(line)

# ── 最优 COALESCE 分布统计 ──────────────────────────────────────
print("\n\n=== 最优 COALESCE 分布 ===")
co_count = {co: 0 for co in COALESCE_LIST}
for batch, totlen in CONFIGS:
    best_co = 1; best_t = 1e9
    for heads in HEADS_LIST:
        for tk in TILE_K_LIST:
            for co in COALESCE_LIST:
                key = (batch, totlen, heads, tk, co)
                if key in results:
                    _, _, total = results[key]
                    if total < best_t:
                        best_t = total
                        best_co = co
    co_count[best_co] += 1
for co in COALESCE_LIST:
    print(f"  CO={co}: {co_count[co]} 个配置")

# ── 按 batch 分组的最优参数 ─────────────────────────────────────
print("\n\n=== 按 batch 分组的最优参数 ===")
for b in sorted(set(c[0] for c in CONFIGS)):
    configs_b = [(bt, tt) for (bt, tt) in CONFIGS if bt == b]
    print(f"\nbatch={b}:")
    print(f"  {'totlen':>6} {'best_h':>6} {'best_tk':>7} {'best_co':>7} {'total_us':>9}")
    for batch, totlen in configs_b:
        best_t = 1e9; best = None
        for heads in HEADS_LIST:
            for tk in TILE_K_LIST:
                for co in COALESCE_LIST:
                    key = (batch, totlen, heads, tk, co)
                    if key in results:
                        _, _, total = results[key]
                        if total < best_t:
                            best_t = total
                            best = (heads, tk, co, total)
        if best:
            h, tk, co, tu = best
            print(f"  {totlen:6d} {h:6d} {tk:7d} {co:7d} {tu:9.1f}")
