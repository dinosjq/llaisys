#!/usr/bin/env python3
import csv, re
from collections import defaultdict

pat6 = re.compile(r'CustomBFloat16, \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+)>')
pat3 = re.compile(r'CustomBFloat16, \(unsigned long\)(\d+), \(unsigned long\)(\d+), \(unsigned long\)(\d+)>')

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,2048),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),(16,256),(16,512),(32,256)]

par6 = defaultdict(list)  # (H,TK,CO,gridX) -> durations
par3 = defaultdict(list)

with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    for row in r:
        n = row[-1]; d = int(row[1]); gx = row[3] if row[3] else ''
        if not gx: continue
        m6 = pat6.search(n)
        if m6:
            H,TK,CO = int(m6.group(2)),int(m6.group(3)),int(m6.group(4))
            par6[(H,TK,CO,int(gx))].append(d/1e3)
            continue
        m3 = pat3.search(n)
        if m3:
            H,TK = int(m3.group(2)),int(m3.group(3))
            par3[(H,TK,int(gx))].append(d/1e3)

def t6(b, t, H, TK, CO):
    bn = t // 64; gx = b * ((bn + CO - 1) // CO)
    lst = par6.get((H,TK,CO,gx), [])
    return lst[1] if len(lst) > 1 else (lst[0] if lst else 0)

def t3(b, t, H, TK):
    bn = t // 64; gx = b * bn
    lst = par3.get((H,TK,gx), [])
    return lst[1] if len(lst) > 1 else (lst[0] if lst else 0)

print(f"{'b×totlen':>14} {'v3_H6T8':>8} {'v6_C1(6,8)':>10} {'v6_C4(6,8)':>10} {'v6_C1(2,16)':>11} {'best':>7} {'cfg':>13} {'v6/v3':>7}")
print("-"*100)
for (b, t) in configs:
    v3 = t3(b, t, 6, 8)
    c1 = t6(b, t, 6, 8, 1)
    c4 = t6(b, t, 6, 8, 4)
    h216 = t6(b, t, 2, 16, 1)
    h316c1 = t6(b, t, 3, 16, 1)
    h316c8 = t6(b, t, 3, 16, 8)
    vals = [(c1,'H6T8C1'),(c4,'H6T8C4'),(h216,'H2T16C1'),(h316c1,'H3T16C1'),(h316c8,'H3T16C8')]
    best, cfg = min(vals, key=lambda x: x[0] if x[0] > 0 else 1e9)
    ratio = f"{best/v3:.2f}x" if v3 else "N/A"
    print(f"  {b:2d}×{t:5d}  {v3:6.1f}  {c1:8.1f}  {c4:8.1f}  {h216:9.1f}  {best:5.1f} {cfg:>12} {ratio}")
