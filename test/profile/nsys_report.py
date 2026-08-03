#!/usr/bin/env python3
"""
Parse nsys SQLite database. Filters only LLAISYS kernels, separates prefill/decode
by timeline pattern, and prints a clean report.

Usage:
  nsys stats --report cuda_gpu_kern_sum results/nsys_qwen2.nsys-rep
  python test/profile/nsys_report.py results/nsys_qwen2.sqlite
"""

import argparse
import os
import sqlite3
import sys
from collections import defaultdict

KERNEL_GROUPS = {
    "paged_attn":     "paged_attention_kernel",
    "flash_decode":   "flash_decoding_parallel_kernel",
    "flash_reduce":   "flash_decoding_reduce_kernel",
    "embedding":      "embedding_gather_kernel",
    "kv_cache_move":  "kv_cache_move_kernel",
    "rms_norm":       "rms_norm_kernel",
    "rope":           "rope_kernel",
    "swiglu":         "swiglu_kernel",
    "add":            "add_kernel<llaisys",
    "add_bias":       "add_bias_kernel<llaisys",
    "topk":           "_topk_kernel<llaisys",
    # cuBLAS names vary by toolkit and matrix dimensions.  Keep these after
    # LLAISYS-specific kernels so phase anchors retain their existing labels.
    "gemm_reduction": "splitKreduce",
    "gemv":           "gemv",
    "gemm":           "gemm",
}

KERNEL_NAMES = {
    "paged_attn":     "paged_attn",
    "flash_decode":   "flash_decode",
    "flash_reduce":   "flash_reduce",
    "embedding":      "embedding",
    "kv_cache_move":  "kv_cache_move",
    "rms_norm":       "rms_norm",
    "rope":           "rope",
    "swiglu":         "swiglu",
    "add":            "add",
    "add_bias":       "linear (bias)",
    "topk":           "topk",
    "gemm_reduction": "linear (reduction)",
    "gemv":           "linear (gemv)",
    "gemm":           "linear (gemm)",
}


def classify(name: str) -> str:
    for label, pat in KERNEL_GROUPS.items():
        if pat in name:
            return label
    return None  # not LLAISYS


def main():
    parser = argparse.ArgumentParser(description="LLAISYS nsys kernel report")
    parser.add_argument("input", help="Path to .sqlite or .nsys-rep file")
    parser.add_argument("--separate-phases", action="store_true", default=True,
                        help="Separate prefill and decode phases (default)")
    parser.add_argument("--no-separate", dest="separate_phases",
                        action="store_false", help="Show all ops combined")
    args = parser.parse_args()

    db_path = args.input
    if db_path.endswith(".nsys-rep") or db_path.endswith(".qdstrm"):
        sqlite_path = db_path.rsplit(".", 1)[0] + ".sqlite"
        if not os.path.exists(sqlite_path):
            import subprocess
            print(f"Exporting {db_path} → {sqlite_path} ...")
            subprocess.run(["nsys", "stats", "--report", "cuda_gpu_kern_sum", db_path],
                           capture_output=True)
        db_path = sqlite_path

    if not os.path.exists(db_path):
        print(f"Error: {db_path} not found.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Query kernels with timestamps
    cur.execute("""
        SELECT s.value, e.start, e.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL e
        JOIN StringIds s ON s.id = e.demangledName
        ORDER BY e.start
    """)
    all_kernels = cur.fetchall()
    conn.close()

    # ── Filter LLAISYS kernels, record timeline ──────────────
    timeline = []
    for name, start, end in all_kernels:
        label = classify(name)
        if label is None:
            continue
        timeline.append((start, end, label, name))

    if not timeline:
        print("No LLAISYS kernels found.")
        sys.exit(1)

    # ── Separate prefill / decode by kernel type ───────────
    # Prefill: contains paged_attention_kernel, no flash_decoding
    # Decode:  contains flash_decoding_parallel_kernel
    prefill_kernels = []
    decode_kernels = []
    current_phase = None
    phase_buf = []

    for s, e, label, name in timeline:
        is_prefill = (label == "paged_attn")
        is_decode  = (label in ("flash_decode", "flash_reduce"))

        if is_prefill and current_phase != "prefill":
            if phase_buf:
                decode_kernels.append(("decode", phase_buf))
            phase_buf = []
            current_phase = "prefill"
        elif is_decode and current_phase != "decode":
            if phase_buf:
                if current_phase == "prefill":
                    prefill_kernels = phase_buf
                else:
                    decode_kernels.append(("decode", phase_buf))
            phase_buf = []
            current_phase = "decode"

        phase_buf.append((s, e, label, name))

    # Don't forget last phase
    if phase_buf:
        if current_phase == "prefill":
            prefill_kernels = phase_buf
        else:
            decode_kernels.append(("decode", phase_buf))

    # Merge all decode phases into one
    all_decode = []
    for _, kerns in decode_kernels:
        all_decode.extend(kerns)

    phases = [("prefill", prefill_kernels)]
    if all_decode:
        phases.append(("decode", all_decode))

    # ── Print report ─────────────────────────────────────────
    t0 = timeline[0][0]
    for phase_label, kernels in phases:
        if not kernels:
            continue
        agg = defaultdict(lambda: {"instances": 0, "total_ns": 0.0,
                                    "min_ns": float("inf"), "max_ns": 0.0})
        for s, e, label, _ in kernels:
            dur = e - s
            entry = agg[label]
            entry["instances"] += 1
            entry["total_ns"] += dur
            entry["min_ns"] = min(entry["min_ns"], dur)
            entry["max_ns"] = max(entry["max_ns"], dur)

        for label in agg:
            if agg[label]["instances"] > 0:
                agg[label]["avg_ns"] = agg[label]["total_ns"] / agg[label]["instances"]

        total = sum(v["total_ns"] for v in agg.values())
        sorted_ops = sorted(agg.items(), key=lambda x: -x[1]["total_ns"])

        # Print phase header
        first_ns = kernels[0][0] - t0
        last_ns = kernels[-1][1] - t0
        print(f"\n  {'=' * 78}")
        print(f"  Phase: {phase_label}  |  "
              f"duration: {(last_ns - first_ns) / 1e6:.2f} ms  |  "
              f"ops: {len(sorted_ops)}  |  kernels: {len(kernels)}")
        print(f"  {'=' * 78}")
        print(f"  {'Operator':<18s} {'Inst':>5s}  {'Total(ms)':>10s}  "
              f"{'Avg(us)':>8s}  {'Min(us)':>8s}  {'Max(us)':>8s}  {'%':>6s}")
        print(f"  {'─' * 78}")

        for label, v in sorted_ops:
            pct = v["total_ns"] / total * 100 if total > 0 else 0
            print(f"  {KERNEL_NAMES[label]:<18s} {v['instances']:5d}  "
                  f"{v['total_ns'] / 1e6:10.3f}  {v['avg_ns'] / 1e3:8.1f}  "
                  f"{v['min_ns'] / 1e3:8.1f}  {v['max_ns'] / 1e3:8.1f}  {pct:5.1f}")

        print(f"  {'─' * 78}")
        print(f"  {'total':<18s} {'':>5s}  {total / 1e6:10.3f}")

    print()


if __name__ == "__main__":
    main()

