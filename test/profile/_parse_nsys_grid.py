#!/usr/bin/env python3
"""解析 nsys cuda_gpu_trace，按顺序关联 (config, params)，提取 parallel kernel duration."""
import csv, re, sys

CSV = "/tmp/v2grid_trace_cuda_gpu_trace.csv"
CONFIGS = [(b, t) for b in [1, 2, 4, 8, 16, 32] for t in [256, 512, 1024, 2048]]
CANDIDATES = [(h, t) for h in [1, 2, 3, 6] for t in [4, 8, 16]]
# 组合顺序: 每 config 先 AUTO 再 12 候选
LABELS = ["AUTO"] + [f"H{h}T{t}" for h, t in CANDIDATES]

# 读取 parallel kernel 行
parallel = []
with open(CSV) as f:
    r = csv.reader(f)
    header = next(r)
    name_idx = header.index("Name")
    dur_idx = header.index("Duration (ns)")
    for row in r:
        name = row[name_idx]
        if "flash_decoding_v2_kernel" in name:
            # 提取 HEADS/TILE_K
            m = re.search(r"kernel<llaisys::\w+, \(unsigned long\)128, \(unsigned long\)(\d+), \(unsigned long\)(\d+)>", name)
            heads = int(m.group(1)) if m else -1
            tile_k = int(m.group(2)) if m else -1
            parallel.append((int(row[dur_idx]), heads, tile_k))

# 每组合 2 个 kernel (预热, 测量)，取第 2 个
results = {}   # (batch, totlen, label) -> duration_us
idx = 0
for batch, totlen in CONFIGS:
    for label in LABELS:
        # 该组合 2 个 parallel kernel
        dur_meas = parallel[idx + 1]   # 第 2 个是测量 (预热在 range 外，测量在 range 内)
        results[(batch, totlen, label)] = dur_meas[0] / 1e3
        idx += 2

# 输出: 每 config 的 AUTO 实际参数 + duration，以及最优候选
print(f"{'batch':>5} {'totlen':>6} {'AUTO(us)':>9} {'AUTO参数':>8} {'best(us)':>9} {'best参数':>9} {'loss':>6}")
tot_loss = 0.0; n = 0
for batch, totlen in CONFIGS:
    auto_us = results[(batch, totlen, "AUTO")]
    auto_heads = -1; auto_tk = -1
    # AUTO 的实际参数: 从 kernel 名，等于该组合第 2 个 kernel 的 heads/tile_k
    # 我们已经跳过预热，用第 idx 之前的结构：重新找
    # (简便：从 candidates 里找与 AUTO 匹配 — 需要 auto 的实际参数，从 raw)
    # 直接重取: 遍历 parallel 找该 config 的 AUTO 测量行
    # 简化: AUTO duration 已有，实际参数在下方打印时跳过
    best_us = 1e18; best_lbl = ""
    for label in LABELS[1:]:
        us = results[(batch, totlen, label)]
        if us < best_us:
            best_us = us; best_lbl = label
    loss = (auto_us - best_us) / best_us * 100
    tot_loss += loss; n += 1
    print(f"{batch:5d} {totlen:6d} {auto_us:9.1f} {'auto':>8} {best_us:9.1f} {best_lbl:>9} {loss:5.1f}%")

print(f"\n平均损失: {tot_loss/n:.1f}%  (仅对比 parallel kernel GPU 耗时)")

# 额外: 打印 AUTO 实际选出的参数 (从 kernel 名模板)
print("\nAUTO 实际参数 (从 kernel 名):")
idx = 0
for batch, totlen in CONFIGS:
    for label in LABELS:
        if label == "AUTO":
            _, heads, tk = parallel[idx + 1]
            print(f"  {batch}x{totlen}: HEADS={heads} TILE_K={tk}")
        idx += 2
