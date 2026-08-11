import csv, re
kernels = []
with open("/tmp/fi_vs_ll_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    si = h.index("Start (ns)"); di = h.index("Duration (ns)")
    for row in r:
        n = row[-1]
        if "BatchDecode" in n or "flash_decoding_v4_kernel" in n or "flash_decoding_v4_reduce" in n:
            kernels.append((int(row[si]), int(row[di]), n))

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),(4,2048),(8,1024),(16,512),(32,256)]

fi_entries = [k for k in kernels if "BatchDecode" in k[2]]
v4_par = [k for k in kernels if "flash_decoding_v4_kernel" in k[2]]
v4_red = [k for k in kernels if "flash_decoding_v4_reduce" in k[2]]

print(f"FI: {len(fi_entries)}  v4 par: {len(v4_par)}  v4 red: {len(v4_red)}")

# Per config: 2 FI (warmup, measured) + 4 v4 (par_warmup, red_warmup, par_meas, red_meas) = 6 entries
print(f"batch x totlen   LLAISYS(us)  FlashInfer(us)  LLAISYS/FlashInfer")
print(f"{'-'*70}")

for ci, (batch, totlen) in enumerate(configs):
    fi_us = fi_entries[ci*2 + 1][1] / 1e3
    base = ci * 4
    pu = v4_par[base + 2][1] / 1e3
    ru = v4_red[base + 2][1] / 1e3
    ll_us = pu + ru
    ratio = fi_us / ll_us if ll_us > 0 else 0
    faster = "LL faster" if ratio > 1 else "FI faster"
    print(f"  {batch:2d} x {totlen:4d}      {ll_us:7.1f}       {fi_us:7.1f}          {ratio:.2f}x  ({faster})")

# mean
ll_sum = 0
for ci, (batch, totlen) in enumerate(configs):
    base = ci * 4
    pu = v4_par[base + 2][1] / 1e3
    ru = v4_red[base + 2][1] / 1e3
    ll_sum += pu + ru
print(f"  mean           {ll_sum/len(configs):7.1f}")
