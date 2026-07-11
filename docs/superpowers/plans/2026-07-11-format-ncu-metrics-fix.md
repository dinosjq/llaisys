# Format Fix & NCU Dynamic Metrics — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix blank-line noise, "#" prefix, broken NCU metrics, --use-ncu table suppression, and non-linear shape selection.

**Architecture:** 4 fixes across 14 files. benchmark_harness.py handles formatting. 12 scripts drop noisy print and gain `--example-index` support in the main loop. ncu_profiler.py gains prefix-based dynamic metric discovery with GPU-keyed caching, and `classify_bottleneck` dynamically looks up metric keys from kernel data.

## Global Constraints

- NCU metric selection: prefix matching with `str.startswith()` — prefixes ending in `.` match all suffixes
- `classify_bottleneck`: search `kernel_data` dict keys dynamically — no hardcoded metric names
- Cache: keyed by GPU model name, 24h TTL, path `results/.ncu_metrics_cache`
- Row numbering: ` 0 ` not `#0 `
- No `print()` in benchmark loops — output only from `save_results()`
- `--use-ncu` branch after `save_results()`, not before benchmark loop
- NCU command includes `--launch-skip` to skip tensor-creation memcpy kernels
- Non-linear scripts: main loop respects `--example-index` to limit shapes
- Benchmark runs twice with `--use-ncu` (first=table, second=NCU). This is inherent to NCU's process-wrapping architecture.

---

### Task 1: Fix row numbering (remove "#" prefix)

**Files:**
- Modify: `test/benchmark/ops/benchmark_harness.py`

- [ ] **Step 1: Change format string**

In `format_results_table()`, three changes:

**Data row** (line ~307):
```python
# Old:
row = (f"  #{i:<3d} {s:<{shape_w}}{_S}...")
# New:
row = (f"  {i:<3d} {s:<{shape_w}}{_S}...")
```

**Header** (line ~296):
```python
# Old:
hdr = (f"  {'# ':<4s}{'shape':<{shape_w}}{_S}...")
# New:
hdr = (f"  {'  ':<3s}{'shape':<{shape_w}}{_S}...")
```

**Sample row** (line ~287):
```python
# Old:
sample = ("#0 " + _S + "x" * shape_w + ...)
# New:
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

```bash
sed -i '/print(format_summary(result))/d' test/benchmark/ops/benchmark_*.py
```

- [ ] **Step 2: Verify**

```bash
grep -n "print(format_summary" test/benchmark/ops/benchmark_*.py && echo "STILL FOUND" || echo "Clean"
for f in test/benchmark/ops/benchmark_*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/benchmark_*.py
git commit -m "fix: remove blank-line noise from benchmark output"
```

---

### Task 3: NCU dynamic metrics + classify fix + cache

**Files:**
- Modify: `test/benchmark/ops/ncu_profiler.py`

- [ ] **Step 1: Replace hardcoded _NCU_METRICS with dynamic discovery**

Delete the module-level `_NCU_METRICS` list. Add these functions before `profile_benchmark()`:

```python
# ---------------------------------------------------------------------------
# dynamic metric discovery (prefix matching via ncu --list-metrics)
# ---------------------------------------------------------------------------

_CACHE_FILE = ".ncu_metrics_cache"

# Prefixes for desired performance metrics.  Those ending with "." are
# prefix-matched against available metric names; those without "." are
# exact-matched.
_DESIRED_PREFIXES = [
    "gpu__time_duration.sum",         # exact — always "sum"
    "sm__throughput.",                 # prefix — sm__throughput.*
    "sm__warps_active.",              # prefix — sm__warps_active.*
    "dram__cycles_elapsed.",          # prefix
    "dram__cycles_active.",           # prefix
]


def _get_available_metrics(ncu_binary: str, results_dir: str) -> list[str]:
    """Return NCU metric names available on this GPU (cached)."""
    import json as _json
    gpu_name = _gpu_model_name()
    cache_path = os.path.join(results_dir, _CACHE_FILE)
    cache: dict = {}
    try:
        if os.path.isfile(cache_path):
            with open(cache_path) as f:
                cache = _json.load(f)
    except Exception:
        pass

    if gpu_name in cache:
        ts, metrics = cache[gpu_name]
        if time.time() - ts < 86400:   # 24-hour TTL
            return metrics

    proc = subprocess.run(
        [ncu_binary, "--list-metrics"],
        capture_output=True, text=True, timeout=30
    )
    if proc.returncode != 0:
        print(f"  WARNING: ncu --list-metrics failed, using empty metric list")
        return []

    metrics = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line and not line.startswith(("-", "=", "ID")):
            parts = line.split()
            if parts:
                metrics.append(parts[0])

    cache[gpu_name] = (time.time(), metrics)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w") as f:
        _json.dump(cache, f)

    return metrics


def _select_metrics(ncu_binary: str, results_dir: str) -> list[str]:
    """Select relevant performance metrics using prefix matching."""
    available = _get_available_metrics(ncu_binary, results_dir)
    selected = []
    for pat in _DESIRED_PREFIXES:
        if pat.endswith("."):
            # prefix match: all metrics starting with this prefix
            matched = [m for m in available if m.startswith(pat)]
            selected.extend(matched)
        else:
            # exact match
            if pat in available:
                selected.append(pat)
    return selected


def _gpu_model_name() -> str:
    """Return a cache-friendly GPU model name."""
    try:
        import torch
        name = torch.cuda.get_device_name(0)
        return name.replace(" ", "_")
    except Exception:
        return "unknown"
```

- [ ] **Step 2: Fix classify_bottleneck to use dynamic keys**

Replace the hardcoded `SM_KEY`, `DRAM_KEY`, `OCC_KEY`, `TIME_KEY` with dynamic lookup from `kernel_data`. The function should search the first kernel entry for matching keys:

```python
def classify_bottleneck(kernel_data: list[dict]) -> dict:
    if not kernel_data:
        return {"bottleneck": "unknown", "sm_throughput_pct": 0.0,
                "dram_throughput_pct": 0.0, "occupancy_pct": 0.0}

    # Dynamically find metric keys in the data
    first = kernel_data[0]
    def _find_key(*patterns: str) -> str:
        for key in first:
            for p in patterns:
                if key.startswith(p):
                    return key
        return ""

    time_key = _find_key("gpu__time_duration.sum")
    sm_key   = _find_key("sm__throughput.")
    dram_key = _find_key("dram__cycles_elapsed.", "dram__cycles_active.",
                         "dram__throughput.")
    occ_key  = _find_key("sm__warps_active.")

    if not time_key:
        return {"bottleneck": "unknown", "sm_throughput_pct": 0.0,
                "dram_throughput_pct": 0.0, "occupancy_pct": 0.0}

    # Weighted aggregation by kernel time
    total_time = sum(k.get(time_key, 0) for k in kernel_data)
    if total_time <= 0:
        return {"bottleneck": "unknown", "sm_throughput_pct": 0.0,
                "dram_throughput_pct": 0.0, "occupancy_pct": 0.0}

    weighted_sm = sum(
        k.get(sm_key, 0.0) * k.get(time_key, 0) for k in kernel_data
    ) / total_time if sm_key else 0.0
    weighted_dram = sum(
        k.get(dram_key, 0.0) * k.get(time_key, 0) for k in kernel_data
    ) / total_time if dram_key else 0.0
    weighted_occ = sum(
        k.get(occ_key, 0.0) * k.get(time_key, 0) for k in kernel_data
    ) / total_time if occ_key else 0.0

    # Classify
    if weighted_dram > 60 and weighted_sm < 20:
        bottleneck = "memory_bound"
    elif weighted_sm > 60 and weighted_dram < 20:
        bottleneck = "compute_bound"
    elif weighted_dram < 30 and weighted_sm < 30:
        bottleneck = "launch_occupancy_bound"
    else:
        bottleneck = "latency_bound"

    return {
        "bottleneck": bottleneck,
        "sm_throughput_pct": weighted_sm,
        "dram_throughput_pct": weighted_dram,
        "occupancy_pct": weighted_occ,
    }
```

- [ ] **Step 3: Update profile_benchmark to use dynamic metrics + --launch-skip**

In `profile_benchmark()`, replace the hardcoded `_NCU_METRICS` reference:

```python
    metrics = _select_metrics(ncu_binary, results_dir)
```

Add `--launch-skip` to the NCU command (skip tensor-creation memcpy kernels):

```python
    ncu_cmd = [ncu_binary,
               "--set", "full",
               "--clock-control", "base",
               "--launch-skip", "5",
               "--target-processes", "all",
               "-o", rep_base,
               "-f",
               "--"] + child_cmd
```

- [ ] **Step 4: Verify syntax and commit**

```bash
python -c "import ast; ast.parse(open('test/benchmark/ops/ncu_profiler.py','r',encoding='utf-8').read()); print('OK')"
git add test/benchmark/ops/ncu_profiler.py
git commit -m "feat: dynamic NCU metrics via prefix matching + dynamic classify_bottleneck + GPU-keyed cache"
```

---

### Task 4: Fix --use-ncu flow + --example-index support in all scripts

**Files:**
- Modify: All 12 `test/benchmark/ops/benchmark_*.py` (except benchmark_harness.py)

**Goal**: (a) `--use-ncu` branch after `save_results()`. (b) Main loop respects `--example-index` to limit shapes. (c) Auto-select from benchmark results.

- [ ] **Step 1: Add --example-index support to main loop of ALL scripts**

For scripts with simple `QWEN2_SHAPES` (add, argmax, embedding, rearrange, rms_norm, rope, swiglu, self_attention, topk, paged_attention, kv_cache_move):

After `args = parser.parse_args()` and repeat limit check, insert:

```python
    # Filter shapes if --example-index is specified
    shapes_to_run = list(enumerate(QWEN2_SHAPES))
    if args.example_index is not None:
        indices = sorted(set(int(p.strip())
            for p in args.example_index.split(",") if p.strip()))
        total = len(QWEN2_SHAPES)
        for idx in indices:
            if idx < 0 or idx >= total:
                print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                sys.exit(1)
        shapes_to_run = [(i, QWEN2_SHAPES[i]) for i in indices]

    results = []
    for idx, shape_info in shapes_to_run:
        # ... existing loop body, using shape_info directly
```

For linear.py: same logic but with `QWEN2_VARIANTS × M_VALUES` flattened index. Convert index → (variant_name, M):

```python
    if args.example_index is not None:
        indices = sorted(set(int(p.strip())
            for p in args.example_index.split(",") if p.strip()))
        total = len(QWEN2_VARIANTS) * len(M_VALUES)
        for idx in indices:
            if idx < 0 or idx >= total:
                print(f"  ERROR: index {idx} out of range (0-{total - 1})")
                sys.exit(1)
        # Build targeted variant/M combinations
        targeted = []
        for idx in indices:
            vi = idx // len(M_VALUES)
            mi = idx % len(M_VALUES)
            vname = list(QWEN2_VARIANTS.keys())[vi]
            targeted.append((vname, QWEN2_VARIANTS[vname], M_VALUES[mi]))
        variants_to_run = []
        ms_to_run = {}
        for vname, (N, K), M in targeted:
            if vname not in ms_to_run:
                ms_to_run[vname] = []
                variants_to_run.append((vname, (N, K)))
            ms_to_run[vname].append(M)
        # Replace the loop with targeted iteration
```

- [ ] **Step 2: Move --use-ncu branch from top to after save_results**

For linear.py:

```python
    save_results(results, args.output)

    if args.use_ncu:
        indices = None
        if args.example_index is None:
            # Auto-select from results
            speeds = [(i, r.speedup) for i, r in enumerate(results)]
            below = [(i, s) for i, s in speeds if s < 1.0]
            if below:
                indices = [min(below, key=lambda x: x[1])[0]]
            else:
                indices = [min(speeds, key=lambda x: abs(x[1] - 1.0))[0]]
        else:
            indices = sorted(set(int(p.strip())
                for p in args.example_index.split(",") if p.strip()))
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
```

For the other 11 scripts: same pattern but `indices` derivation from `results`:

```python
    save_results(results, args.output)

    if args.use_ncu:
        indices = None
        if args.example_index is None:
            speeds = [(i, r.speedup) for i, r in enumerate(results)]
            below = [(i, s) for i, s in speeds if s < 1.0]
            if below:
                indices = [min(below, key=lambda x: x[1])[0]]
            else:
                indices = [min(speeds, key=lambda x: abs(x[1] - 1.0))[0]]
        else:
            indices = sorted(set(int(p.strip())
                for p in args.example_index.split(",") if p.strip()))
        from ncu_profiler import profile_benchmark
        profile_benchmark(script=__file__, argv=sys.argv, op_name=OP_NAME,
                          example_indices=indices)
```

- [ ] **Step 3: Update profile_benchmark signature to remove results param**

Remove the `results` parameter from `profile_benchmark()`. The function receives only `script`, `argv`, `op_name`, and `example_indices`. Auto-select is done in the benchmark script (Task 4 Step 2).

Also remove `_auto_select_index()` — no longer needed.

- [ ] **Step 4: Verify syntax for all scripts**

```bash
for f in test/benchmark/ops/benchmark_*.py; do
  python -c "import ast; ast.parse(open('$f','r',encoding='utf-8').read())" || echo "FAIL: $f"
done
echo "All OK"
```

- [ ] **Step 5: Commit in groups**

```bash
# linear.py (most complex, has variant flattening)
git add test/benchmark/ops/benchmark_linear.py test/benchmark/ops/ncu_profiler.py
git commit -m "fix: --use-ncu after save_results + --example-index in main loop + dynamic NCU"

# simple ops
git add test/benchmark/ops/benchmark_{add,argmax,embedding,rearrange}.py
git commit -m "fix: --use-ncu after save_results + --example-index in simple op benchmarks"

# norm/activation ops
git add test/benchmark/ops/benchmark_{rms_norm,rope,swiglu}.py
git commit -m "fix: --use-ncu after save_results + --example-index in norm/activation benchmarks"

# attention/topk ops
git add test/benchmark/ops/benchmark_{self_attention,topk}.py
git commit -m "fix: --use-ncu after save_results + --example-index in attention/topk benchmarks"

# paged/cache ops
git add test/benchmark/ops/benchmark_{paged_attention,kv_cache_move}.py
git commit -m "fix: --use-ncu after save_results + --example-index in paged/cache benchmarks"
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

- [ ] **Step 2: Run benchmark — verify clean output**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --warmup 2 --repeat 3
```
Expected: no blank lines, row numbers without `#`.

- [ ] **Step 3: Run --use-ncu — verify table + NCU**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --use-ncu --example-index "3" --warmup 2 --repeat 3
```
Expected: table output first, then NCU profiling with non-zero metrics.

- [ ] **Step 4: Run add benchmark — verify non-linear path**

```bash
python test/benchmark/ops/benchmark_add.py --dtype f16 --warmup 2 --repeat 3
```
Expected: clean table, 6 rows with numbers.
