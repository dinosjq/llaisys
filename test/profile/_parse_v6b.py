#!/usr/bin/env python3
"""Parse v6 bench by sequential order — each config runs: 2 v3 + 5 v5 + 5 v6 = 12 calls.
Each call = par(4: warmup+meas) ... actually launch has par+reduce = 4 kernel entries per call (2 per kernel).
"""
import csv, re

kernels = []
with open("/tmp/v6_bench_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    for row in r:
        n = row[-1]; d = int(row[1])
        if 'flash_decoding' in n and ('_kernel' in n or '_reduce' in n):
            ver = re.search(r'flash_decoding_v(\d)', n).group(1)
            # extract HEADS,TK,CO
            m = re.search(r'\(unsigned long\)128, \(unsigned long\)(\d+), \(unsigned long\)(\d+)(?:, \(unsigned long\)(\d+))?\)', n)
            kernels.append((int(d)/1e3, int(ver), n))

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,2048),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),(16,256),(16,512),(32,256)]

# Per config: 2 v3 calls + 5 v5 + 5 v6 = 12 calls, each call = 2 kernels (par+reduce)
# 12 calls * 4 kernel-entries (warmup+measured par, warmup+measured red) = 48
# Structure per call: par_warmup, red_warmup, par_measured, red_measured
# So per config: 48 kernel entries. First 8 = v3 (2 calls), next 20 = v5, next 20 = v6

per = 48
total = len(configs) * per

# But order: v3(2 calls=8 entries), then v5(5 calls=20), then v6(5 calls=20)
print(f"Total entries needed: {total}, got: {len(kernels)}")

# For each config, entries are grouped: v3 x2, v5 x5, v6 x5 calls
# but kernel name version tells us which version. Let's filter by version.
v6_kernels = [k for k in kernels if k[1] == 6]
v3_kernels = [k for k in kernels if k[1] == 3]

print(f"v3 entries: {len(v3_kernels)}, v6 entries: {len(v6_kernels)}")

# v3: 2 calls per config = 8 entries per config
# v6: 5 calls per config = 20 entries per config
def get_call_total(entries, call_idx):
    """entries are per-call: par_w, red_w, par_m, red_m. call_idx starts at 0."""
    base = call_idx * 4
    return entries[base+2][0] + entries[base+3][0]

print(f"\n{'b×totlen':>14} {'v3_H6T8':>8} | {'v6 C1(6,8)':>10} {'C4(6,8)':>9} {'C4(3,16)':>9} {'best':>7} {'cfg':>12} {'v6/v3':>7}")
for ci, (b, t) in enumerate(configs):
    v3_base = ci * 8
    v3_h6 = get_call_total(v3_kernels, v3_base // 4)  # call 0 = H6T8
    v3_h2 = get_call_total(v3_kernels, (v3_base + 4) // 4)  # call 1 = H2T16

    v6_base = ci * 20
    # 5 calls: (6,8,1),(6,8,4),(3,16,4),(2,16,1),(3,16,8)
    v6_vals = [get_call_total(v6_kernels, (v6_base + ti*4)//4) for ti in range(5)]
    c1, c4, c316c4, c216c1, c316c8 = v6_vals
    best = min(v6_vals)
    cfgs = ['H6T8C1','H6T8C4','H3T16C4','H2T16C1','H3T16C8']
    bcfg = cfgs[v6_vals.index(best)]
    ratio = f"{best/v3_h6:.2f}x" if v3_h6 else "N/A"
    print(f"  {b:2d}×{t:5d}  {v3_h6:6.1f} | {c1:8.1f} {c4:7.1f} {c316c4:7.1f} {best:5.1f} {bcfg:>11} {ratio}")
