#!/usr/bin/env python3
import csv, re

v3_par = re.compile(r"flash_decoding_v3_kernel"); v3_red = re.compile(r"flash_decoding_v3_reduce")
v5_par = re.compile(r"flash_decoding_v5_kernel"); v5_red = re.compile(r"flash_decoding_v5_reduce")
v6_par = re.compile(r"flash_decoding_v6_kernel"); v6_red = re.compile(r"flash_decoding_v6_reduce")
# extract template params from mangled name: (HD, HEADS, TILE_K, COALESCE)
par_pat = re.compile(r"flash_decoding_v(\d)_kernel<.*?\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+),\s*\(unsigned long\)(\d+)\)")

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,2048),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),(16,256),(16,512),(32,256)]

# Parse all kernels with template params
par_by_cfg = {}  # (version, batch, totlen, H, TK, CO) -> list of par durations
red_by_cfg = {}
with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r); ni = h.index("Name"); di = h.index("Duration (ns)")
    for row in r:
        n = row[ni]; d = int(row[di])
        m = par_pat.search(n)
        if not m: continue
        ver = int(m.group(1)); HD, H, TK, CO = int(m.group(2)), int(m.group(3)), int(m.group(4)), int(m.group(5))
        # Find which config this belongs to by gridX: grid.x = tot_task_num = batch*ceil(bn/CO)*...
        # Actually easier: track order. We know sequence: for each config, v3(2), v5(5), v6(5)
        # Use gridX = tot_task_num to infer (batch, totlen) via block count
        gx = int(row[3])
        # tot_task_num = batch * ceil(bn / CO), bn = totlen/64
        # find matching config
        for (b, t) in configs:
            bn = t // 64
            ttn = b * ((bn + CO - 1) // CO)
            if ttn == gx:
                key = (ver, b, t, H, TK, CO)
                par_by_cfg.setdefault(key, []).append(d/1e3)
                break

# Group: take 2nd occurrence (measured) per key
print(f"{'b×totlen':>14} {'v3_H6T8':>8} {'v6_H6T8C1':>9} {'v6_H6T8C4':>9} {'v6_H3T16C4':>10} {'v6_best':>8} {'cfg':>12} {'v6/v3':>7}")
for (b, t) in configs:
    bn = t // 64
    v3k = (3, b, t, 6, 8, 1)
    v3_list = par_by_cfg.get(v3k, [])
    v3_t = v3_list[1] if len(v3_list) > 1 else (v3_list[0] if v3_list else 0)

    # v6 candidates
    v6_vals = []
    for H, TK, CO in [(6,8,1),(6,8,4),(3,16,4),(2,16,1),(3,16,8)]:
        k = (6, b, t, H, TK, CO)
        lst = par_by_cfg.get(k, [])
        if len(lst) > 1:
            v6_vals.append((lst[1], H, TK, CO))

    v6_68c1 = next((v for v,H,TK,CO in v6_vals if H==6 and TK==8 and CO==1), 0)
    v6_68c4 = next((v for v,H,TK,CO in v6_vals if H==6 and TK==8 and CO==4), 0)
    v6_316c4 = next((v for v,H,TK,CO in v6_vals if H==3 and TK==16 and CO==4), 0)
    if v6_vals:
        best, bH, bTK, bCO = min(v6_vals)
        ratio = f"{best/v3_t:.2f}x" if v3_t else "N/A"
        print(f"  {b:2d}×{t:5d}  {v3_t:6.1f}  {v6_68c1:7.1f}  {v6_68c4:7.1f}  {v6_316c4:8.1f}  {best:6.1f}  H{bH}T{bTK}C{bCO}  {ratio}")
