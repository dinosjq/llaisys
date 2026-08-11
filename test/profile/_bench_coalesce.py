#!/usr/bin/env python3
"""探索 coalesce（合并连续 KV 块为 task）对 flash_decoding 性能的影响.
KV cache 布局不变 (TN=64)，只改 grid.x = ceil(tot_block_num / coalesce).
通过 nsys 精确测 parallel kernel GPU 时间."""
import torch, sys, os, ctypes, itertools, re, subprocess, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys

NH, NKVH, HD, TN = 12, 2, 128, 64
DEV = torch.device("cuda:0")
BV2 = 2
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [
    c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3

CONFIGS = [(b, t) for b in [1, 4, 8, 16, 32] for t in [256, 512, 1024, 2048, 4096]]
COALESCE = [1, 2, 4, 8]

# nsys profile 脚本
script = os.path.join(os.path.dirname(__file__), '_run_coalesce.py')

# 生成 profile target
with open(script, 'w') as f:
    f.write("""#!/usr/bin/env python3
import torch, sys, os, ctypes
sys.path.insert(0, r'{test}')
sys.path.insert(0, r'{python}')
from ctypes import c_void_p, c_size_t, c_float, c_int
import llaisys
lib = ctypes.CDLL(llaisys.libllaisys.LIB_LLAISYS._name)
lib.llaisysFlashDecodingV3.argtypes = [c_void_p]*10 + [c_size_t]*4 + [c_size_t, c_int, c_float] + [c_size_t]*4 + [c_int]*3
NH,NKVH,HD,TN,BV2=12,2,128,64,2; DEV=torch.device('cuda:0')
configs = {configs}
coalesce_list = {coalesce}
for batch,totlen in configs:
    for c in coalesce_list:
        bn = (totlen//TN + c - 1)//c * c  # round up to nearest c*TN
        actual_bn = totlen//TN
        mb = bn + 1
        # build block_ids with block-level prefix (same as original)
        tbids = torch.zeros((batch, mb), dtype=torch.int64, device=DEV)
        for b in range(batch):
            tbids[b, 0] = (b+1)*actual_bn
            tbids[b, 1:actual_bn+1] = torch.arange(actual_bn, device=DEV)
        qt = torch.randn((batch,NH,HD), dtype=torch.bfloat16, device=DEV)
        kt = torch.randn((max(actual_bn,1), TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
        vt = torch.randn((max(actual_bn,1), TN, NKVH, HD), dtype=torch.bfloat16, device=DEV)
        av = torch.empty((batch,NH,HD), dtype=torch.bfloat16, device=DEV)
        cut = torch.arange(batch+1, dtype=torch.int64, device=DEV)
        tot = torch.full((batch,), totlen, dtype=torch.int64, device=DEV)
        tb = batch * actual_bn
        acc = torch.zeros((tb,NH,HD), dtype=torch.float32, device=DEV)
        asum = torch.zeros((tb,NH,1), dtype=torch.float32, device=DEV)
        amax = torch.zeros((tb,NH,1), dtype=torch.float32, device=DEV)
        s = 1.0/(HD**0.5)
        nparts = (actual_bn + c - 1)//c
        tot_task = batch * nparts
        args = (av.data_ptr(),acc.data_ptr(),asum.data_ptr(),amax.data_ptr(),
                qt.data_ptr(),kt.data_ptr(),vt.data_ptr(),tbids.data_ptr(),cut.data_ptr(),tot.data_ptr(),
                TN, batch, mb, tot_task, 1, BV2, s, NH, HD, HD, NKVH, 6, 8, c)
        label = f"c{batch}x{totlen}_CO{c}"
        # warmup + measured
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_push(label)
        lib.llaisysFlashDecodingV3(*args); torch.cuda.synchronize()
        torch.cuda.nvtx.range_pop()
""".format(test=os.path.join(os.path.dirname(__file__), '..'),
           python=os.path.join(os.path.dirname(__file__), '..', '..', 'python'),
           configs=CONFIGS, coalesce=COALESCE))

# 跑 nsys
print("Running nsys profile...")
subprocess.run(["nsys", "profile", "-t", "cuda,nvtx", "-o", "/tmp/coalesce", "--force-overwrite=true",
                "python", script], check=True, timeout=300, capture_output=True)
print("Exporting trace...")
subprocess.run(["rm", "-f", "/tmp/coalesce.sqlite", "/tmp/coalesce_t*.csv"], check=False)
subprocess.run(["nsys", "stats", "--report=cuda_gpu_trace", "--format=csv",
                "-o", "/tmp/coalesce_t", "/tmp/coalesce.nsys-rep"], check=True, timeout=120, capture_output=True)

# 解析
print("\n=== coalesce 扫描 (parallel kernel us) ===")
import csv as cs
rows = []
with open("/tmp/coalesce_t_cuda_gpu_trace.csv") as f:
    r = cs.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        if "flash_decoding_v3_kernel" in row[ni]:
            rows.append(int(row[di])/1e3)
# 每 config+coalesce: 2 个 parallel (warmup+meas), 取第 2 个
print(f"{'batch':>5} {'totlen':>6}  ", end="")
for c in COALESCE: print(f"CO={c:>7}", end="")
print("  best_co")
idx = 0
for batch, totlen in CONFIGS:
    vals = {}
    for c in COALESCE:
        vals[c] = rows[idx*2+1]; idx += 1
    best = min(vals, key=vals.get)
    print(f"{batch:5d} {totlen:6d}  ", end="")
    for c in COALESCE: print(f"{vals[c]:7.1f}", end="")
    print(f"  CO={best}")
