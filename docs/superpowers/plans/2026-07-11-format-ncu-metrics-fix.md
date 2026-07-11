# Format Fix & NCU Dynamic Metrics — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix blank-line noise, "#" prefix, broken NCU metrics, and --use-ncu table suppression.

**Architecture:** Four independent fixes across 14 files. benchmark_harness.py handles formatting; 12 benchmark scripts drop noisy print; ncu_profiler.py gains dynamic metric discovery.

**Tech Stack:** Python 3.9+, ncu CLI

## Global Constraints

- No hardcoded NCU metric names — use prefix matching against `ncu --list-metrics` output
- Row numbering: `0  ` not `#0 `
- No `print()` in benchmark loops — output only from `save_results()`
- `--use-ncu` runs benchmark first, then profiles

---

### Task 1: Fix row numbering (remove "#" prefix)

**Files:**
- Modify: `test/benchmark/ops/benchmark_harness.py`

- [ ] **Step 1: Change format string**

In `format_results_table()`, change:

```python
row = (f"  #{i:<3d} {s:<{shape_w}}{_S}{r.dtype:<{dtype_w}}{_S}"
```

To:

```python
row = (f"  {i:<3d} {s:<{shape_w}}{_S}{r.dtype:<{dtype_w}}{_S}"
```

And header:

```python
hdr = (f"  {'# ':<4s}{'shape':<{shape_w}}{_S}..."
```

To:

```python
hdr = (f"  {'  ':<3s}{'shape':<{shape_w}}{_S}..."
```

And sample row:

```python
sample = ("#0 " + _S + "x" * shape_w + ...)
```

To:

```python
sample = ("  0 " + _S + "x" * shape_w + ...)
```

- [ ] **Step 2: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/benchmark_harness.py','r',encoding='utf-8').read()); print('OK')"
git add test/benchmark/ops/benchmark_harness.py
git commit -m "fix: remove # prefix from table row numbering"
```

---

### Task 2: Remove blank lines from benchmark loops

**Files:**
- Modify: All 12 `test/benchmark/ops/benchmark_*.py` (except benchmark_harness.py)

- [ ] **Step 1: Delete `print(format_summary(result))` from all scripts**

For each script, find and delete the line `print(format_summary(result))` inside the main loop:

```bash
sed -i '/print(format_summary(result))/d' test/benchmark/ops/benchmark_*.py
```

- [ ] **Step 2: Also delete `print(format_summary(result))` call from linear.py**

Check benchmark_linear.py has it in the variant/M loop:

```bash
grep -n "print(format_summary" test/benchmark/ops/benchmark_linear.py
```
If found, remove it.

- [ ] **Step 3: Verify all scripts parse**

```bash
for f in test/benchmark/ops/benchmark_*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 4: Commit**

```bash
git add test/benchmark/ops/benchmark_*.py
git commit -m "fix: remove blank-line noise from benchmark output"
```

---

### Task 3: NCU dynamic metrics discovery

**Files:**
- Modify: `test/benchmark/ops/ncu_profiler.py`

- [ ] **Step 1: Add metric discovery function**

Add to `ncu_profiler.py` before `profile_benchmark()`:

```python
# ---------------------------------------------------------------------------
# dynamic metric discovery
# ---------------------------------------------------------------------------

_METRICS_CACHE_FILE = "results/ncu/.metrics_cache"


def _get_available_metrics(ncu_binary: str, cache_dir: str = "") -> list[str]:
    """Return all NCU metric names available on this GPU.

    Uses ``ncu --list-metrics``, caching the result to avoid repeated
    subprocess calls.
    """
    import json as _json
    cache_path = os.path.join(cache_dir or os.path.dirname(__file__),
                              _METRICS_CACHE_FILE)
    try:
        if os.path.isfile(cache_path):
            mtime = os.path.getmtime(cache_path)
            if time.time() - mtime < 86400:  # 24-hour cache
                with open(cache_path) as f:
                    return _json.load(f)
    except Exception:
        pass

    proc = subprocess.run(
        [ncu_binary, "--list-metrics"],
        capture_output=True, text=True, timeout=30
    )
    metrics = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith(("-", "=", "ID")):
            name = line.split()[0] if line.split() else ""
            if name:
                metrics.append(name)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(metrics, f)

    return metrics


def _select_metrics(ncu_binary: str, cache_dir: str = "") -> list[str]:
    """Select relevant performance metrics from those available locally."""
    available = _get_available_metrics(ncu_binary, cache_dir)

    # desired prefixes in priority order
    desired = [
        "gpu__time_duration.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "dram__cycles_elapsed.sum",
        "dram__cycles_active.avg",
    ]

    selected = []
    for prefix in desired:
        matched = [m for m in available if m == prefix]
        if matched:
            selected.append(matched[0])
    return selected
```

- [ ] **Step 2: Update profile_benchmark() to use dynamic metrics**

In `profile_benchmark()`, replace the hardcoded `_NCU_METRICS` with:

```python
    # Dynamic metric selection (cached)
    metrics = _select_metrics(ncu_binary, results_dir)
```

Replace all uses of `_NCU_METRICS` with the local `metrics` variable.

Delete the module-level `_NCU_METRICS` list.

- [ ] **Step 3: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/ncu_profiler.py','r',encoding='utf-8').read()); print('OK')"
git add test/benchmark/ops/ncu_profiler.py
git commit -m "feat: dynamic NCU metric discovery via ncu --list-metrics"
```

---

### Task 4: Restructure --use-ncu flow

**Files:**
- Modify: All 12 `test/benchmark/ops/benchmark_*.py` (except benchmark_harness.py)

**Current (broken):**
```python
if args.use_ncu:
    profile_benchmark(...); return
# ... benchmark loop + save_results
```

**Fixed:**
```python
# ... benchmark loop + save_results
if args.use_ncu:
    profile_benchmark(...)
```

- [ ] **Step 1: Move --use-ncu branch after save_results in all scripts**

For each script, move the `if args.use_ncu:` block from the top of main() to after `save_results()`. In benchmark_linear.py this is:

```python
    save_results(results, args.output)

    if args.use_ncu:
        # ... (same profile_benchmark call as before)
```

For the other 11 scripts:

```python
    save_results(results, args.output)

    if args.use_ncu:
        indices = None
        if args.example_index is not None:
            indices = sorted(set(
                int(p.strip()) for p in args.example_index.split(",") if p.strip()))
            total = len(QWEN2_SHAPES)
            for idx in indices:
                if idx < 0 or idx >= total:
                    print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                    sys.exit(1)
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
```

**Key difference from old position**: The `profile_benchmark` call now uses `results` data for auto-select, instead of re-running the benchmark.

For linear.py specifically, update `profile_benchmark` call to accept results for auto-select:

```python
    if args.use_ncu:
        indices = None
        if args.example_index is not None:
            indices = sorted(set(
                int(p.strip()) for p in args.example_index.split(",") if p.strip()))
        else:
            # Auto-select from results: worst speedup
            speeds = [(i, r.speedup) for i, r in enumerate(results)]
            below = [(i, s) for i, s in speeds if s < 1.0]
            if below:
                idx = min(below, key=lambda x: x[1])[0]
            else:
                idx = min(speeds, key=lambda x: abs(x[1] - 1.0))[0]
            indices = [idx]
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
```

- [ ] **Step 2: Update profile_benchmark() to accept pre-computed results**

Add optional `results` parameter to `profile_benchmark()`:

```python
def profile_benchmark(script: str, argv: list[str], op_name: str,
                      example_indices: Optional[list[int]] = None,
                      results: Optional[list] = None) -> None:
```

When `results` is provided and `example_indices` is None, auto-select from the pre-computed results instead of re-running the benchmark.

- [ ] **Step 3: Verify syntax**

```bash
for f in test/benchmark/ops/benchmark_*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 4: Commit**

```bash
# linear first (complex, has auto-select from results)
git add test/benchmark/ops/benchmark_linear.py test/benchmark/ops/ncu_profiler.py
git commit -m "fix: move --use-ncu after benchmark output, pass results for auto-select"

# remaining 11 scripts
git add test/benchmark/ops/benchmark_{add,argmax,embedding,rearrange,rms_norm,rope,swiglu,self_attention,topk,kv_cache_move,paged_attention}.py
git commit -m "fix: move --use-ncu after save_results in all benchmark scripts"
```

---

### Task 5: Final verification

- [ ] **Step 1: All files parse**

```bash
for f in test/benchmark/ops/*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 2: Run benchmark to verify clean output**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --warmup 2 --repeat 3
```

Expected: no blank lines between `Correctness: PASS` and table. Row numbering without `#`.

- [ ] **Step 3: Run --use-ncu to verify table + NCU**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --use-ncu --example-index "3" --warmup 2 --repeat 3
```

Expected: table output first, then NCU profiling with non-zero metrics.
