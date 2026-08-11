#!/usr/bin/env python3
import json
with open("/tmp/v5_results.json") as f:
    raw = json.load(f)
results = {}
for k, v in raw.items():
    results[tuple(eval(k))] = v[2]

configs = [(1,256),(1,512),(1,1024),(1,2048),(1,4096),
           (2,256),(2,512),(2,1024),(2,2048),
           (4,256),(4,512),(4,1024),(4,2048),
           (8,256),(8,512),(8,1024),
           (16,256),(16,512),(32,256)]
HEADS = [1,2,3,6]; TKS = [8,16]; COS = [1,2,3,4,6,8]

v3_base = {(1,256):11.4,(1,512):11.2,(1,1024):15.4,(1,2048):23.5,(1,4096):41.4,
           (2,256):11.1,(2,512):15.7,(2,1024):25.3,(2,2048):39.5,
           (4,256):16.2,(4,512):25.7,(4,1024):39.5,(4,2048):70.0,
           (8,256):27.2,(8,512):42.9,(8,1024):70.7,
           (16,256):46.0,(16,512):73.8,(32,256):80.5}

print(f"{'bxtotlen':>14} {'v3':>7} {'v5_best':>8} {'v5/v3':>7} {'best_cfg':>17} | CO effect (H=3,TK=16): CO1..CO8")
print(f"{'':>14} {'':>7} {'us':>8} {'':>7} {'(H,TK,CO)':>17} |")
print("-" * 120)
for b,t in configs:
    v3 = v3_base.get((b,t), 0)
    best = 1e9; best_p = None
    for h in HEADS:
        for tk in TKS:
            for co in COS:
                k = (b,t,h,tk,co)
                if k in results and results[k] < best:
                    best = results[k]; best_p = (h,tk,co)
    co_str = ""
    for co in COS:
        k = (b,t,3,16,co)
        if k in results:
            co_str += f" {results[k]:6.1f}"
    h,tk,co = best_p
    ratio = f"{best/v3:.2f}x" if v3 > 0 else "N/A"
    print(f"  {b:2d} x {t:5d}  {v3:6.1f}  {best:7.1f}  {ratio:>6}  H={h} TK={tk:2d} CO={co:2d}       |{co_str}")

# Also show H=6,TK=8 CO breakdown
print(f"\n{'bxtotlen':>14} | CO effect (H=6,TK=8): CO1..CO8")
for b,t in configs:
    co_str = ""
    for co in COS:
        k = (b,t,6,8,co)
        if k in results:
            co_str += f" {results[k]:6.1f}"
    print(f"  {b:2d} x {t:5d}  |{co_str}")

# Best config for each batch size summary
print(f"\nSummary by batch:")
print(f"{'batch':>6} | best cfg for small totlen | best cfg for large totlen")
for b in [1,2,4,8,16,32]:
    configs_b = [(bt,tt) for (bt,tt) in configs if bt == b]
    if len(configs_b) >= 2:
        small_t = configs_b[0]; large_t = configs_b[-1]
        best_s = min(((results.get((b,small_t[1],h,tk,co), 1e9), h, tk, co)
                      for h in HEADS for tk in TKS for co in COS), key=lambda x: x[0])
        best_l = min(((results.get((b,large_t[1],h,tk,co), 1e9), h, tk, co)
                      for h in HEADS for tk in TKS for co in COS), key=lambda x: x[0])
        print(f"  {b:4d}   | H={best_s[1]} TK={best_s[2]} CO={best_s[3]:2d} ({best_s[0]:.1f}us)  | H={best_l[1]} TK={best_l[2]} CO={best_l[3]:2d} ({best_l[0]:.1f}us)")
