#!/usr/bin/env python3
"""Batch runner: execute all operator benchmarks and aggregate results."""

import json, os, subprocess, sys, time

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))
OPERATORS = ["add", "argmax", "embedding", "rearrange", "linear",
             "rms_norm", "rope", "swiglu", "self_attention", "topk"]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all operator benchmarks")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--op", default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--ncu", nargs="?", const="detailed", choices=["quick", "detailed", "full"])
    parser.add_argument("--output", default=os.path.join(BENCHMARK_DIR, "results"))
    args = parser.parse_args()

    ops_to_run = OPERATORS if args.op == "all" else args.op.split(",")

    print(f"=== LLAISYS Operator Benchmarks ===")
    print(f"    dtype={args.dtype}  warmup={args.warmup}  repeat={args.repeat}")
    print(f"    ops: {', '.join(ops_to_run)}")
    print()

    failed = []
    for op in ops_to_run:
        script = os.path.join(BENCHMARK_DIR, f"benchmark_{op}.py")
        if not os.path.exists(script):
            print(f"  SKIP {op} (no benchmark script)")
            continue
        cmd = [sys.executable, script, "--dtype", args.dtype,
               "--warmup", str(args.warmup), "--repeat", str(args.repeat),
               "--output", args.output]
        if args.ncu:
            cmd += ["--ncu", args.ncu]
        print(f"  Running {op}...")
        start = time.time()
        ret = subprocess.run(cmd, capture_output=False)
        elapsed = time.time() - start
        if ret.returncode != 0:
            print(f"  FAIL {op} (exit code {ret.returncode})")
            failed.append(op)
        else:
            print(f"  DONE {op} ({elapsed:.1f}s)")

    if failed:
        print(f"\n  FAILED: {', '.join(failed)}")
        sys.exit(1)
    print(f"\n  All benchmarks complete. Results: {args.output}/latest.json")

if __name__ == "__main__":
    main()
