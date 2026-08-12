#!/usr/bin/env python3
import csv, re

# Template params: HD, HEADS, TILE_K, COALESCE after CustomBFloat16
pat = re.compile(r"flash_decoding_v6_kernel<.*?, \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+)\)")

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,2048),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),(16,256),(16,512),(32,256)]

# Collect per (gridX, H, TK, CO) all par durations
from collections import defaultdict
par_by = defaultdict(list)
with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    for row in r:
        n = row[-1]; d = int(row[1]); gxs = row[3]
        if not gxs: continue
        gx = int(gxs)
        if 'flash_decoding_v6_kernel' in n:
            m = pat.search(n)
            if m:
                H, TK, CO = int(m.group(2)), int(m.group(3)), int(m.group(4))
                # gridX = tot_task_num = batch * ceil(bn/CO). bn = totlen/64
                for (b, t) in configs:
                    bn = t // 64
                    ttn = b * ((bn + CO - 1) // CO)
                    if ttn == gx:
                        par_by[(b, t, H, TK, CO)].append(d/1e3)
                        break

# Also get v3 for baseline
v3_pat = re.compile(r"flash_decoding_v3_kernel<.*?, \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+)\)")
v3_by = defaultdict(list)
with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    for row in r:
        n = row[-1]; d = int(row[1]); gxs = row[3]
        if not gxs: continue
        gx = int(gxs)
        if 'flash_decoding_v3_kernel' in n:
            m = v3_pat.search(n)
            if m:
                H, TK = int(m.group(2)), int(m.group(3))
                for (b, t) in configs:
                    bn = t // 64
                    ttn = b * bn
                    if ttn == gx:
                        v3_by[(b, t, H, TK)].append(d/1e3)
                        break

print(f"{'b×totlen':>14} {'v3_H6T8':>8} | {'v6_H6T8C1':>9} {'C4':>7} {'C1(3,16)':>8} {'best':>7} {'cfg':>12} {'v6/v3':>7}")
for (b, t) in configs:
    v3_list = v3_by.get((b, t, 6, 8), [])
    v3_t = v3_list[1] if len(v3_list) > 1 else (v3_list[0] if v3_list else 0)

    def pick(H, TK, CO):
        lst = par_by.get((b, t, H, TK, CO), [])
        return lst[1] if len(lst) > 1 else (lst[0] if lst else 0)

    v6_c1 = pick(6, 8, 1)
    v6_c4 = pick(6, 8, 4)
    v6_316c1 = pick(3, 16, 1)
    v6_216c1 = pick(2, 16, 1)
    v6_316c8 = pick(3, 16, 8)
    vals = [(v6_c1,'H6T8C1'),(v6_c4,'H6T8C4'),(v6_316c1,'H3T16C1'),(v6_216c1,'H2T16C1'),(v6_316c8,'H3T16C8')]
    best, cfg = min(vals, key=lambda x: x[0] if x[0] > 0 else 1e9)
    ratio = f"{best/v3_t:.2f}x" if v3_t else "N/A"
    print(f"  {b:2d}×{t:5d}  {v3_t:6.1f} | {v6_c1:7.1f} {v6_c4:5.1f} {v6_316c1:6.1f} {best:5.1f} {cfg:>11} {ratio}")
