# Remove --ncu-child, implement --use-ncu — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `--ncu` / `--ncu-child` two-flag system with single `--use-ncu` flag backed by centralized `profile_benchmark()` in ncu_profiler.py. Remove all NCU logic from individual benchmark scripts.

**Architecture:** Each benchmark script has a 3-line `--use-ncu` branch at the top of main() that delegates to `ncu_profiler.profile_benchmark()`. That function re-launches the script under NCU as a normal benchmark run limited to one shape, then parses the `.ncu-rep` and saves results. No hidden flags. No NCU knowledge in benchmark scripts beyond the delegation.

**Tech Stack:** Python 3.9+, torch, llaisys ctypes, ncu CLI

## Global Constraints

- `--repeat` default 10, maximum 20
- `--set full` only (no choice)
- `--example-index` comma-separated, 0-based, validated against actual shape count, deduplicated
- Auto-select: speedup < 1.0 → min speedup (worst); all >= 1.0 → speedup closest to 1.0
- NCU errors: check returncode + print stderr + check .ncu-rep existence
- NCU results: independent `ncu_results.json` alongside the `.ncu-rep`
- Table rows numbered `#0, #1, ...` in first column
- `ncu_profile.sh` deleted
- No hidden flags; subprocess is a normal benchmark run
- CUDA only (no CPU benchmarks)

---

### Task 1: benchmark_harness.py — row numbering + repeat limit

**Files:**
- Modify: `test/benchmark/ops/benchmark_harness.py`

**Interfaces:**
- Produces: `format_results_table(results)` now prints `#N` before each row
- Produces: `run_benchmark` unchanged

- [ ] **Step 1: Add row numbering to format_results_table**

Add `#N` as first column. Modify the data row section of `format_results_table()`:

```python
# in format_results_table(), replace the data-row loop:

for i, r in enumerate(results):
    s = _shape_str(r)
    row = (f"  #{i:<3d} {s:<{shape_w}}{_S}{r.dtype:<{dtype_w}}{_S}"
           f"{r.latency_us:{val_w}.1f}{_S}{r.TFLOPS:{val_w}.1f}")
    if has_baseline:
        if r.baseline_latency_us > 0:
            row += (f"{_S}{r.baseline_latency_us:{bl_w}.1f}{_S}"
                    f"{r.baseline_TFLOPS:{bl_w}.1f}{_S}"
                    f"{r.speedup:{sp_w}.2f}x")
        else:
            row += (f"{_S}{'—':>{bl_w}}{_S}{'—':>{bl_w}}{_S}"
                    f"{'—':>{sp_w}}")
    lines.append(row)
    latencies.append(r.latency_us)
```

Also update the header to add `#  ` prefix, and widen `W` accordingly (add `_S + "xxx"` to sample).

Sample row computation — add `_S + "xxx"` (for `#0 ` prefix of 4 chars):

```python
sample = (_S + "xxx" + _S + "x" * shape_w + _S + "x" * dtype_w +
          _S + "x" * val_w + _S + "x" * val_w)
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/benchmark_harness.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/benchmark_harness.py
git commit -m "feat: add row numbering (#N) to benchmark output table"
```

---

### Task 2: ncu_profiler.py — add profile_benchmark(), remove unused

**Files:**
- Modify: `test/benchmark/ops/ncu_profiler.py`

**Interfaces:**
- Produces: `profile_benchmark(script: str, argv: list[str], op_name: str, example_indices: list[int] | None = None) -> None`
- Keeps: `find_ncu()`, `extract_metrics_from_report()`, `classify_bottleneck()`, `NCUConfig`, `ProfileReport`
- Keeps as-is: `parse_ncu_csv()`, `run_under_ncu()`

- [ ] **Step 1: Add profile_benchmark() function**

Add to `ncu_profiler.py`:

```python
def profile_benchmark(script: str, argv: list[str], op_name: str,
                      example_indices: Optional[list[int]] = None) -> None:
    """Run a benchmark script under NCU for selected shapes.

    1. If example_indices is None/empty, runs the full benchmark first
       to collect speedup data, then auto-selects the worst-performing
       shape (lowest speedup) to profile.
    2. Re-launches the SAME script under NCU, limited to the selected
       shape via --M / --variant arguments.
    3. Parses the .ncu-rep, classifies the bottleneck, saves
       ncu_results.json.

    Parameters
    ----------
    script : str
        Path to the benchmark script (e.g. __file__ from caller).
    argv : list[str]
        Original sys.argv from the caller.
    op_name : str
        Operator name for output file naming (e.g. "linear").
    example_indices : list[int] | None
        Specific shape indices to profile.  None = auto-select.
    """
    import json, time

    # --- build child args (strip --use-ncu and --example-index) ---
    stripped = []
    skip_next = False
    for a in argv:
        if skip_next:
            skip_next = False
            continue
        if a == "--use-ncu":
            continue
        if a == "--example-index":
            skip_next = True
            continue
        if a.startswith("--example-index="):
            continue
        stripped.append(a)

    # --- determine which shape(s) to profile ---
    ncu_binary = find_ncu()
    out_dir = os.path.join(os.path.dirname(script), "results", "ncu")
    os.makedirs(out_dir, exist_ok=True)

    results_dir = os.path.join(os.path.dirname(script), "results")
    ncu_results = []

    for idx in (example_indices or [None]):
        shape_args = _build_shape_args(idx, stripped, script)
        child_cmd = [sys.executable, script] + shape_args

        rep_file = os.path.join(out_dir, f"{op_name}_idx{idx}.ncu-rep" if idx is not None
                                else f"{op_name}.ncu-rep")

        ncu_cmd = [ncu_binary,
                   "--set", "full",
                   "--clock-control", "base",
                   "--target-processes", "all",
                   "-o", rep_file.replace(".ncu-rep", ""),
                   "-f",
                   "--"] + child_cmd

        proc = subprocess.run(ncu_cmd, capture_output=True, text=True, timeout=600)

        if proc.returncode != 0:
            print(f"NCU failed (exit {proc.returncode}):\n{proc.stderr}")
            sys.exit(1)

        if not os.path.isfile(rep_file):
            print(f"NCU report not generated: {rep_file}\n{proc.stderr}")
            sys.exit(1)

        # --- parse ---
        metrics = [
            "gpu__time_duration.sum",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            "dram__throughput.avg.pct_of_peak_sustained_elapsed",
            "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
        ]
        raw = extract_metrics_from_report(ncu_binary, rep_file, metrics)
        classif = classify_bottleneck(raw)

        total_time_ns = sum(k.get("gpu__time_duration.sum", 0) for k in raw)
        entry = {
            "operator": op_name,
            "shape_index": idx,
            "report_file": rep_file,
            "kernel_count": len(raw),
            "total_time_us": total_time_ns / 1000.0,
            "bottleneck": classif["bottleneck"],
            "sm_throughput_pct": classif["sm_throughput_pct"],
            "dram_throughput_pct": classif["dram_throughput_pct"],
            "occupancy_pct": classif["occupancy_pct"],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ncu_results.append(entry)
        print(f"  NCU #{idx}: {classif['bottleneck']}  "
              f"SM={classif['sm_throughput_pct']:.1f}%  "
              f"DRAM={classif['dram_throughput_pct']:.1f}%  "
              f"Occ={classif['occupancy_pct']:.1f}%")

    # --- save ---
    json_path = os.path.join(results_dir, "ncu_results.json")
    with open(json_path, "w") as f:
        json.dump(ncu_results, f, indent=2)
    print(f"  NCU results saved: {json_path}")


def _build_shape_args(index: int | None, argv: list[str], script: str) -> list[str]:
    """Build CLI args to run the benchmark on a specific shape index.

    If index is None: auto-select via running the benchmark and picking
    the worst speedup.
    """
    if index is None:
        # Run benchmark once to collect speedup data
        import json as _json, tempfile, subprocess as _sp
        tmp = os.path.join(tempfile.gettempdir(), f"_ncu_auto_{os.getpid()}.json")
        auto_cmd = [sys.executable, script] + [a for a in argv
                    if not a.startswith("--use-ncu") and not a.startswith("--example-index")]
        # Force a clean run: no --use-ncu, output to temp
        auto_cmd += ["--output", os.path.dirname(tmp)]
        _sp.run(auto_cmd, capture_output=True)

        # Read latest.json from temp output
        latest = os.path.join(os.path.dirname(tmp), "latest.json")
        if not os.path.isfile(latest):
            # No results — fall back to index 0
            return _shape_args_for_index(0, argv, script)
        with open(latest) as f:
            data = _json.load(f)
        results = data.get("results", [])
        if not results:
            return _shape_args_for_index(0, argv, script)

        # Find worst speedup
        speeds = [(i, r.get("speedup", 1.0)) for i, r in enumerate(results)]
        below_1 = [(i, s) for i, s in speeds if s < 1.0]
        if below_1:
            idx = min(below_1, key=lambda x: x[1])[0]  # lowest speedup = biggest gap
        else:
            idx = min(speeds, key=lambda x: abs(x[1] - 1.0))[0]  # closest to 1.0

    return _shape_args_for_index(index if index is not None else idx, argv, script)


def _shape_args_for_index(index: int, argv: list[str], script: str) -> list[str]:
    """Return args to run script on a single shape by index."""
    # Each benchmark script stores its shapes in a predictable structure.
    # For linear: 3 variants x 6 M values = 18 shapes
    # For others: shapes list with 6 entries
    # We pass --example-index to the script which the benchmark must handle
    # by extracting shape info from the results table.
    #
    # But the child process needs concrete --M and --variant args.
    # The cleanest approach: run the script once to get shape metadata,
    # then map index → CLI args.

    # For simplicity, assume linear.py structure first:
    M_VALUES = [1, 16, 64, 128, 512, 2048]
    VARIANTS = ["q_proj", "gate_proj", "down_proj"]

    # Detect if this is benchmark_linear.py
    script_name = os.path.basename(script)
    if script_name == "benchmark_linear.py":
        total = len(VARIANTS) * len(M_VALUES)
        if index >= total:
            raise ValueError(f"Index {index} out of range (0-{total-1})")
        variant_idx = index // len(M_VALUES)
        m_idx = index % len(M_VALUES)
        return ["--variant", VARIANTS[variant_idx], "--M", str(M_VALUES[m_idx])]
    else:
        # Non-linear scripts: pass the index directly; the benchmark
        # script will select the shape by index from QWEN2_SHAPES
        return ["--example-index", str(index)]
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/ncu_profiler.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/ncu_profiler.py
git commit -m "feat: add profile_benchmark() — centralized NCU profiling entry point"
```

---

### Task 3: benchmark_linear.py — full refactoring

**Files:**
- Modify: `test/benchmark/ops/benchmark_linear.py`

**Interfaces:**
- Consumes: `profile_benchmark` from `ncu_profiler`
- Produces: `--use-ncu`, `--example-index` CLI args; row-numbered output

- [ ] **Step 1: Replace argparse NCU args**

Replace:
```python
parser.add_argument("--ncu", nargs="?", const="detailed", choices=["quick", "detailed", "full"])
parser.add_argument("--ncu-child", action="store_true", help=argparse.SUPPRESS)
```

With:
```python
parser.add_argument("--use-ncu", action="store_true",
                    help="Profile with NCU (NVIDIA Nsight Compute, --set full)")
parser.add_argument("--example-index", type=str, default=None,
                    help="Comma-separated shape indices to profile (0-based)")
parser.add_argument("--repeat", type=int, default=10,
                    help="Timed iterations per shape (max 20)")
```

- [ ] **Step 2: Add --use-ncu branch at top of main()**

Right after `args = parser.parse_args()`:

```python
    # --use-ncu: re-launch under NCU, then exit
    if args.use_ncu:
        indices = None
        if args.example_index is not None:
            indices = []
            for part in args.example_index.split(","):
                part = part.strip()
                if part:
                    indices.append(int(part))
            # deduplicate
            indices = sorted(set(indices))
            # validate upper bound
            total_shapes = len(QWEN2_VARIANTS) * len(M_VALUES)
            for idx in indices:
                if idx < 0 or idx >= total_shapes:
                    print(f"  ERROR: index {idx} out of range (0-{total_shapes-1})")
                    sys.exit(1)
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
        return
```

- [ ] **Step 3: Add --repeat limit validation**

After args parsing, before the loop:

```python
    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20
```

- [ ] **Step 4: Remove old NCU blocks**

Delete:
- `if args.ncu and not args.ncu_child:` block (lines ~128-141)
- `if args.ncu_child:` block (lines ~143-146)
- The lazy import `from ncu_profiler import NCUConfig, profile_with_ncu`

- [ ] **Step 5: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/benchmark_linear.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 6: Commit**

```bash
git add test/benchmark/ops/benchmark_linear.py
git commit -m "refactor: replace --ncu/--ncu-child with --use-ncu in benchmark_linear.py"
```

---

### Task 4: Remaining 11 benchmark scripts — batch refactoring

**Files:**
- Modify: All `test/benchmark/ops/benchmark_*.py` except `benchmark_linear.py` and `benchmark_harness.py`

**Pattern for each script** — identical to Task 3 with operator-specific adjustments:

- [ ] **Step 1: For each script, replace argparse**

Change:
```python
parser.add_argument("--ncu", ...)
parser.add_argument("--ncu-child", ...)
```
To:
```python
parser.add_argument("--use-ncu", action="store_true",
                    help="Profile with NCU (NVIDIA Nsight Compute, --set full)")
parser.add_argument("--example-index", type=str, default=None,
                    help="Comma-separated shape indices to profile (0-based)")
```

And change `--repeat` default from 100 to 10.

- [ ] **Step 2: Add --use-ncu branch at top of main()**

For scripts without `--variant` (add, argmax, embedding, rearrange, rms_norm, rope, swiglu, self_attention, topk, paged_attention, kv_cache_move):

```python
    if args.use_ncu:
        indices = None
        if args.example_index is not None:
            indices = sorted(set(
                int(p.strip()) for p in args.example_index.split(",") if p.strip()))
            total = len(QWEN2_SHAPES)
            for idx in indices:
                if idx < 0 or idx >= total:
                    print(f"  ERROR: index {idx} out of range (0-{total-1})")
                    sys.exit(1)
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
        return
```

- [ ] **Step 3: Add --repeat limit**

```python
    if args.repeat > 20:
        print(f"  WARNING: --repeat capped at 20 (requested {args.repeat})")
        args.repeat = 20
```

- [ ] **Step 4: Remove old NCU blocks from each script**

Delete:
- `if args.ncu and not args.ncu_child:` block
- `if args.ncu_child:` block  
- `from ncu_profiler import NCUConfig, profile_with_ncu` (lazy import inside the NCU block)

- [ ] **Step 5: Verify all files parse**

```bash
for f in test/benchmark/ops/benchmark_{add,argmax,embedding,rearrange,rms_norm,rope,swiglu,self_attention,topk,kv_cache_move,paged_attention}.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read()); print('$f: OK')"
done
```

- [ ] **Step 6: Commit each group**

```bash
# Group 1: simple ops (add, argmax, embedding, rearrange)
git add test/benchmark/ops/benchmark_{add,argmax,embedding,rearrange}.py
git commit -m "refactor: replace --ncu/--ncu-child with --use-ncu in simple op benchmarks"

# Group 2: norm/activation ops (rms_norm, rope, swiglu)
git add test/benchmark/ops/benchmark_{rms_norm,rope,swiglu}.py
git commit -m "refactor: replace --ncu/--ncu-child with --use-ncu in norm/activation benchmarks"

# Group 3: attention/topk ops (self_attention, topk)
git add test/benchmark/ops/benchmark_{self_attention,topk}.py
git commit -m "refactor: replace --ncu/--ncu-child with --use-ncu in attention/topk benchmarks"

# Group 4: paged/cache ops (paged_attention, kv_cache_move)
git add test/benchmark/ops/benchmark_{paged_attention,kv_cache_move}.py
git commit -m "refactor: replace --ncu/--ncu-child with --use-ncu in paged/cache benchmarks"
```

---

### Task 5: Delete ncu_profile.sh

- [ ] **Step 1: Delete**

```bash
git rm test/benchmark/ops/ncu_profile.sh
```

- [ ] **Step 2: Commit**

```bash
git commit -m "chore: remove ncu_profile.sh (replaced by --use-ncu)"
```

---

### Task 6: Final verification

- [ ] **Step 1: All Python files parse**

```bash
for f in test/benchmark/ops/*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())"
done && echo "All OK"
```

- [ ] **Step 2: Verify no NCU references remain in benchmark scripts**

```bash
grep -l "ncu_child\|args\.ncu\|NCUConfig\|profile_with_ncu" test/benchmark/ops/benchmark_*.py && echo "WARNING: stale NCU refs found" || echo "Clean"
```

- [ ] **Step 3: Run a benchmark to verify output**

```bash
python test/benchmark/ops/benchmark_add.py --dtype f16 --warmup 2 --repeat 5
```
Expected: row-numbered table, no blank lines.

- [ ] **Step 4: Run --use-ncu if NCU available**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --use-ncu --M 128 --variant q_proj --warmup 2 --repeat 5
```
Expected: NCU profiling, bottleneck printed, `ncu_results.json` created.
