import csv, re
kernels = []
with open("/tmp/fi_vs_v4_t_cuda_gpu_trace.csv") as f:
    r = csv.reader(f); h = next(r)
    si = h.index("Start (ns)"); di = h.index("Duration (ns)")
    for row in r:
        name = row[-1]
        if "flash_decoding_v4_kernel" in name or "flash_decoding_v4_reduce" in name or "BatchDecode" in name:
            kernels.append((int(row[si]), int(row[di]), name))

# Configs: 7 (batch,totlen) pairs
configs = [(1,2048),(4,2048),(8,2048),(16,2048),(1,4096),(4,4096),(16,4096)]
# Per config: 1 FI call + 7 v4 calls = 8 calls
# Each call: warmup measured via nvtx

# Classify entries
fi_entries = [k for k in kernels if "BatchDecode" in k[2]]
v4_par_entries = [k for k in kernels if "flash_decoding_v4_kernel" in k[2]]
v4_red_entries = [k for k in kernels if "flash_decoding_v4_reduce" in k[2]]

print("=== FlashInfer vs LLAISYS v4 (NH=8,NKVH=2,HD=128) ===")
print(f"FI kernels: {len(fi_entries)}  V4 par: {len(v4_par_entries)}  V4 red: {len(v4_red_entries)}")

# Per config: FI has 2 entries (warmup, measured). v4 has 7 calls × 2 entries (par+red)
# Pattern: FI warmup, FI measured, then for each v4 combo: (par warmup, red warmup, par measured, red measured)

tests = [(6,8,1),(6,8,2),(6,8,4),(6,8,8),(3,16,1),(3,16,8),(2,16,1)]

for ci, (batch, totlen) in enumerate(configs):
    # FI: entries ci*2 and ci*2+1
    fi_us = fi_entries[ci*2+1][1] / 1e3  # measured
    
    print(f"\nbatch={batch} totlen={totlen}:")
    print(f"  FlashInfer: {fi_us:.1f}us")
    
    # v4 combos
    for ti, (h,tk,co) in enumerate(tests):
        base = ci * len(tests) * 4 + ti * 4
        if base + 3 < len(v4_par_entries):
            pu = v4_par_entries[base+2][1]/1e3  # measured par
            ru = v4_red_entries[base+2][1]/1e3  # measured red
            print(f"  v4 H={h} TK={tk} CO={co}: par={pu:.1f}us red={ru:.1f}us total={pu+ru:.1f}us  vs FI={fi_us/(pu+ru):.2f}x")
