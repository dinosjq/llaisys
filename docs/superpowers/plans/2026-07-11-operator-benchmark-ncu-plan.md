# Operator Performance Benchmark & NCU Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CUDA operator performance benchmarking system under `test/benchmark/ops/` with NCU profiling integration, using Qwen2 model-driven shapes.

**Architecture:** Shared harness (`benchmark_harness.py`) provides CUDA-event timing + stats; NCU module (`ncu_profiler.py`) provides Python subprocess double-launch + CSV parsing; each operator gets a standalone `benchmark_<op>.py` script defining its shapes and reference implementation; `run_all_benchmarks.py` orchestrates batch execution.

**Tech Stack:** Python 3.9+, PyTorch (for CUDA events + reference impls), LLAISYS ctypes bindings, NCU 2022.2+

**Key Design Decision:** Use `torch.cuda.Event` for GPU timing (LLAISYS RuntimeAPI has no CUDA event functions). Timing controls GPU execution time only — not CPU launch overhead.

## Global Constraints

- CUDA (NVIDIA) only; no CPU benchmarks
- Shapes from Qwen2 model: M ∈ {1, 16, 64, 128, 512, 2048}, N/K from actual layer dims (hs=3584, di=18944, nh=28, nkvh=4, dh=128)
- L2 cache defeat via buffer rotation (3 sets)
- Report min as primary metric (CUTLASS convention) + full stats distribution
- NCU three-tier: `--ncu quick` (default set), `--ncu detailed`, `--ncu full`
- No baseline comparison for `kv_cache_move` and `paged_attention`
- JSON output per run + terminal summary
- Reuse `test/test_utils.py` for `random_tensor`, `random_int_tensor`, `zero_tensor`

---

## File Layout

```
test/benchmark/ops/
├── __init__.py                    # empty marker
├── benchmark_harness.py           # shared: CUDATimer, BenchmarkConfig, BenchmarkResult, run_benchmark()
├── ncu_profiler.py                # shared: NCUConfig, find_ncu(), run_under_ncu(), parse_ncu_csv(), classify_bottleneck()
├── benchmark_add.py               # simple ops with PyTorch CUDA ref
├── benchmark_argmax.py
├── benchmark_embedding.py
├── benchmark_rearrange.py
├── benchmark_linear.py            # cuBLAS ref, multiple (N,K) pairs
├── benchmark_rms_norm.py          # Unsloth Triton ref
├── benchmark_rope.py              # Unsloth fused QK-RoPE ref
├── benchmark_swiglu.py            # Unsloth fused SwiGLU ref
├── benchmark_self_attention.py    # Flash-Attention v2 Triton ref
├── benchmark_topk.py              # PyTorch CUDA ref
├── benchmark_paged_attention.py   # no baseline comparison
├── benchmark_kv_cache_move.py     # no baseline comparison
├── run_all_benchmarks.py          # batch runner + JSON aggregation
├── ncu_profile.sh                 # shell convenience wrapper
└── results/                       # output directory (gitignored)
    ├── latest.json
    └── ncu/*.ncu-rep
```

## Import Convention

All benchmark scripts are in `test/benchmark/ops/`. They import from `test/` via:

```python
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import llaisys
import torch
from test_utils import random_tensor, random_int_tensor, zero_tensor, check_equal
from benchmark_harness import BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary
from ncu_profiler import NCUConfig, find_ncu, run_under_ncu, parse_ncu_csv, classify_bottleneck, ProfileReport
```

---

### Task 1: Directory structure and __init__.py

**Files:**
- Create: `test/benchmark/ops/__init__.py`

**Interfaces:**
- Produces: empty file, marks the package

- [ ] **Step 1: Create directory and __init__.py**

```bash
mkdir -p test/benchmark/ops test/benchmark/ops/results/ncu
touch test/benchmark/ops/__init__.py
```

- [ ] **Step 2: Commit**

```bash
git add test/benchmark/ops/__init__.py
git commit -m "chore: create benchmark/ops directory structure"
```

---

### Task 2: benchmark_harness.py — shared timing framework

**Files:**
- Create: `test/benchmark/ops/benchmark_harness.py`

**Interfaces:**
- Produces:
  - `CUDATimer` class: `record_start()`, `record_end()`, `elapsed_ms()` → float, `destroy()`
  - `BenchmarkConfig` dataclass: `op_name: str`, `shape: dict`, `dtype_name: str`, `warmup: int = 10`, `repeat: int = 100`
  - `BenchmarkResult` dataclass: `operator`, `dtype`, `shape`, `latency_us`, `latency_stats`, `bandwidth_GBs`, `TFLOPS`
  - `run_benchmark(config, setup_fn, llsys_fn, ref_fn=None)` → `BenchmarkResult`
  - `format_summary(result)` → str (terminal output)
  - `save_results(results, path)` → None (writes JSON + prints terminal summary)

- [ ] **Step 1: Implement CUDATimer using torch.cuda.Event**

Write `test/benchmark/ops/benchmark_harness.py`:

```python
"""Shared benchmark harness for CUDA operator performance testing.

Uses torch.cuda.Event for GPU-side timing (LLAISYS RuntimeAPI has no
CUDA event API). Reports min as primary metric per CUTLASS convention.
"""

from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Optional

import torch


# ── CUDA Event Timer ───────────────────────────────────────────

class CUDATimer:
    """GPU-side timer using torch.cuda.Event for accurate kernel timing."""

    def __init__(self):
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._stream = torch.cuda.current_stream()

    def record_start(self) -> None:
        self._start.record(self._stream)

    def record_end(self) -> None:
        self._end.record(self._stream)

    def elapsed_ms(self) -> float:
        """Block until the end event completes, then return elapsed ms."""
        self._end.synchronize()
        return self._start.elapsed_time(self._end)


# ── Buffer Pool for L2 Cache Defeat ─────────────────────────────

class BufferPool:
    """Rotate through N buffer sets to prevent L2 cache from inflating
    measured bandwidth. On each iteration we pick the next set so that
    by the time a set is reused its data has been evicted from cache."""

    def __init__(self, n_sets: int, alloc_fn: Callable[[], dict]):
        if n_sets < 2:
            raise ValueError("BufferPool needs at least 2 buffer sets")
        self._sets = [alloc_fn() for _ in range(n_sets)]
        self._cursor = 0

    def next(self) -> dict:
        """Return (and rotate to) the next buffer set."""
        bufs = self._sets[self._cursor]
        self._cursor = (self._cursor + 1) % len(self._sets)
        return bufs

    def __len__(self) -> int:
        return len(self._sets)


# ── Configuration & Result Types ─────────────────────────────────

@dataclass
class BenchmarkConfig:
    op_name: str
    shape: dict          # e.g. {"M": 512, "N": 3584, "K": 3584}
    dtype_name: str      # "f16" | "bf16" | "f32"
    warmup: int = 10
    repeat: int = 100
    total_bytes_read: int = 0    # set by setup_fn for bandwidth calc
    total_bytes_write: int = 0
    total_flops: int = 0         # set by setup_fn for TFLOPS calc


@dataclass
class BenchmarkResult:
    operator: str
    dtype: str
    shape: dict
    # LLAISYS metrics
    latency_us: float            # min across iterations
    latency_stats: dict          # {min, median, mean, stddev, p99}
    bandwidth_GBs: float
    TFLOPS: float
    # Baseline (optional)
    baseline_name: str = ""
    baseline_latency_us: float = 0.0
    baseline_TFLOPS: float = 0.0
    speedup: float = 0.0         # baseline / llaissys (>1 means baseline is faster)
    # NCU (optional, merged after profiling)
    ncu_set: str = ""
    ncu_report_file: str = ""
    ncu_bottleneck: str = ""
    ncu_sm_throughput_pct: float = 0.0
    ncu_dram_throughput_pct: float = 0.0
    ncu_occupancy_pct: float = 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        # keep raw_times compact
        return d

    @staticmethod
    def compute_stats(times_us: list[float]) -> dict:
        """CUTLASS convention: report min as the primary metric.
        Noise sources only add latency — min is closest to hardware limit."""
        if not times_us:
            return {}
        sorted_times = sorted(times_us)
        n = len(sorted_times)
        return {
            "min": sorted_times[0],
            "median": statistics.median(sorted_times),
            "mean": statistics.mean(sorted_times),
            "stddev": statistics.stdev(sorted_times) if n >= 2 else 0.0,
            "p99": sorted_times[int(n * 0.99)] if n >= 100 else sorted_times[-1],
        }


# ── Core Benchmark Loop ─────────────────────────────────────────

def run_benchmark(
    config: BenchmarkConfig,
    setup_fn: Callable[[], dict],        # () -> {"in": Tensor, "out": Tensor, ...}
    llsys_fn: Callable[[dict], None],    # (buffers) -> None  (launches LLAISYS kernel)
    ref_fn: Callable[[dict], None] = None,  # (buffers) -> None  (launches reference kernel)
) -> BenchmarkResult:
    """Run a full warmup + repeat benchmark cycle.

    setup_fn is called once per buffer set during warmup and repeat phases.
    Each invocation must return a dict of LLAISYS Tensors. The same dict
    keys are passed to llsys_fn / ref_fn.

    Returns a BenchmarkResult with LLAISYS metrics. If ref_fn is provided,
    baseline fields are populated.
    """
    # ── Warmup phase ──
    for _ in range(config.warmup):
        buffers = setup_fn()
        llsys_fn(buffers)

    # Ensure warmup kernels finish before timed section
    torch.cuda.synchronize()

    # ── Timed phase (LLAISYS) ──
    timer = CUDATimer()
    llsys_times_us = []

    for _ in range(config.repeat):
        buffers = setup_fn()
        timer.record_start()
        llsys_fn(buffers)
        timer.record_end()
        elapsed = timer.elapsed_ms()
        llsys_times_us.append(elapsed * 1000.0)  # ms -> us

    stats = BenchmarkResult.compute_stats(llsys_times_us)
    latency_us = stats["min"]
    elapsed_s = latency_us / 1e6

    # Derived metrics
    bandwidth_GBs = 0.0
    if config.total_bytes_read + config.total_bytes_write > 0 and elapsed_s > 0:
        total_gb = (config.total_bytes_read + config.total_bytes_write) / 1e9
        bandwidth_GBs = total_gb / elapsed_s

    tflops = 0.0
    if config.total_flops > 0 and elapsed_s > 0:
        tflops = config.total_flops / elapsed_s / 1e12

    result = BenchmarkResult(
        operator=config.op_name,
        dtype=config.dtype_name,
        shape=config.shape,
        latency_us=latency_us,
        latency_stats=stats,
        bandwidth_GBs=bandwidth_GBs,
        TFLOPS=tflops,
    )

    # ── Baseline ──
    if ref_fn is not None:
        # Extra warmup for reference
        for _ in range(config.warmup):
            buffers = setup_fn()
            ref_fn(buffers)
        torch.cuda.synchronize()

        ref_times_us = []
        for _ in range(config.repeat):
            buffers = setup_fn()
            timer.record_start()
            ref_fn(buffers)
            timer.record_end()
            ref_times_us.append(timer.elapsed_ms() * 1000.0)

        ref_min_us = min(ref_times_us)
        ref_elapsed_s = ref_min_us / 1e6
        ref_tflops = 0.0
        if config.total_flops > 0 and ref_elapsed_s > 0:
            ref_tflops = config.total_flops / ref_elapsed_s / 1e12

        result.baseline_name = "reference"
        result.baseline_latency_us = ref_min_us
        result.baseline_TFLOPS = ref_tflops
        result.speedup = ref_min_us / latency_us if latency_us > 0 else 0.0

    return result


# ── Helpers ─────────────────────────────────────────────────────

def elem_size(dtype_name: str) -> int:
    return {"f16": 2, "bf16": 2, "f32": 4, "i64": 8, "i32": 4}.get(dtype_name, 4)


def format_summary(result: BenchmarkResult) -> str:
    """Human-readable single-result summary."""
    shape_str = "  ".join(f"{k}={v}" for k, v in result.shape.items())
    lines = [
        f"  {result.operator}  {result.dtype}  {shape_str}",
        f"  {'─' * 55}",
        f"    LLAISYS:  {result.latency_us:8.1f} us  │  {result.TFLOPS:7.1f} TFLOPS  │  {result.bandwidth_GBs:7.1f} GB/s",
    ]
    if result.baseline_name:
        lines.append(
            f"    {result.baseline_name:8s}:  {result.baseline_latency_us:8.1f} us  │  "
            f"{result.baseline_TFLOPS:7.1f} TFLOPS"
        )
        lines.append(f"    {'─' * 55}")
        tag = "faster" if result.speedup >= 1.0 else "slower"
        pct = abs(result.speedup - 1.0) * 100
        lines.append(f"    Speedup: {result.speedup:.2f}x  (LLAISYS is {pct:.0f}% {tag} than {result.baseline_name})")
    if result.ncu_bottleneck:
        lines.append(
            f"    NCU:  {result.ncu_bottleneck}  SM={result.ncu_sm_throughput_pct:.0f}%  "
            f"DRAM={result.ncu_dram_throughput_pct:.0f}%  Occ={result.ncu_occupancy_pct:.0f}%"
        )
    return "\n".join(lines)


def save_results(results: list[BenchmarkResult], output_dir: str = "test/benchmark/ops/results") -> str:
    """Write results to JSON file and print terminal summary.

    Returns the path to the written JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)

    import datetime
    ts = datetime.datetime.now().strftime("%Y-%m-%d/%H%M%S")
    date_dir = os.path.join(output_dir, ts)
    os.makedirs(date_dir, exist_ok=True)
    json_path = os.path.join(date_dir, "results.json")

    # Environment metadata
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    env = {
        "cuda_version": torch.version.cuda,
        "gpu_model": gpu_name,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    payload = {
        "environment": env,
        "results": [r.to_dict() for r in results],
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)

    # Symlink latest
    latest_link = os.path.join(output_dir, "latest.json")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.unlink(latest_link)
    os.symlink(os.path.abspath(json_path), latest_link)

    # Terminal summary
    print(f"\n{'=' * 70}")
    print(f"  Benchmark Results  |  {gpu_name}  |  CUDA {torch.version.cuda}")
    print(f"{'=' * 70}")
    for r in results:
        print(format_summary(r))
        print()
    print(f"  Results saved to: {json_path}")
    print(f"  NCU reports:      {os.path.join(output_dir, 'ncu')}")

    return json_path
```

- [ ] **Step 2: Verify imports work**

```bash
cd test/benchmark/ops && python -c "
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..', '..')))
from benchmark_harness import CUDATimer, BenchmarkConfig, BenchmarkResult, run_benchmark
print('benchmark_harness imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/benchmark_harness.py
git commit -m "feat: add benchmark harness with CUDA event timing and buffer rotation"
```

---

### Task 3: ncu_profiler.py — NCU integration module

**Files:**
- Create: `test/benchmark/ops/ncu_profiler.py`

**Interfaces:**
- Produces:
  - `NCUConfig` dataclass: `set`, `clock_control`, `launch_skip`, `launch_count`, `kernel_name`, `metrics`
  - `find_ncu()` → str (path to ncu binary, raises FileNotFoundError if absent)
  - `run_under_ncu(ncu_binary, app_cmd, config, output_file)` → `subprocess.CompletedProcess`
  - `parse_ncu_csv(csv_text: str, metrics: list[str])` → list[dict] (one dict per kernel row)
  - `classify_bottleneck(kernel_data: list[dict])` → dict with `bottleneck`, `sm_throughput_pct`, `dram_throughput_pct`, `occupancy_pct`
  - `ProfileReport` dataclass
  - `profile_with_ncu(benchmark_script: str, op_args: list, ncu_config: NCUConfig, output_dir: str)` → `ProfileReport`

- [ ] **Step 1: Implement ncu_profiler.py**

Write `test/benchmark/ops/ncu_profiler.py`:

```python
"""NCU (NVIDIA Nsight Compute) integration for operator profiling.

Uses the Python subprocess double-launch pattern (Karpathy llm.c style):
  1. Benchmark script launched without --ncu-child runs normally and
     detects --ncu flag → re-launches itself under ncu as a subprocess.
  2. The child process (with --ncu-child) runs the kernel once; NCU
     captures hardware counters and writes a .ncu-rep file.
  3. The parent process reads the .ncu-rep, exports metrics as CSV,
     parses them, and reports a bottleneck classification.
"""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional


# ── Configuration ───────────────────────────────────────────────

@dataclass
class NCUConfig:
    set: str = "detailed"               # default | detailed | full
    clock_control: str = "base"         # lock clocks for reproducibility
    launch_skip: int = 10               # skip warmup kernel launches
    launch_count: int = 1               # profile only 1 launch after skip
    kernel_name: Optional[str] = None   # regex to filter kernels
    replay_mode: str = "kernel"         # kernel | application | range
    metrics: list = field(default_factory=lambda: [
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ])

    def to_cli_args(self) -> list[str]:
        args = [
            "--set", self.set,
            "--replay-mode", self.replay_mode,
            "--launch-skip", str(self.launch_skip),
            "--launch-count", str(self.launch_count),
            "--clock-control", self.clock_control,
            "--target-processes", "all",
        ]
        if self.kernel_name:
            args += ["--kernel-name", self.kernel_name]
        return args


@dataclass
class ProfileReport:
    operator: str
    ncu_set: str
    report_file: str
    kernel_count: int
    # Aggregated across all profiled kernels
    total_time_us: float
    sm_throughput_pct: float       # weighted by time
    dram_throughput_pct: float
    occupancy_pct: float
    bottleneck: str                # memory_bound | compute_bound | launch_occupancy_bound | latency_bound
    raw_metrics: list[dict]        # parsed CSV rows


# ── NCU Binary Discovery ────────────────────────────────────────

def find_ncu() -> str:
    """Locate the ncu binary. Checks PATH first, then common install paths.

    On WSL, also checks the Windows-side Nsight Compute installation.
    """
    ncu = shutil.which("ncu")
    if ncu is not None:
        return ncu

    candidates = [
        "/usr/local/cuda/bin/ncu",
        "/opt/cuda/bin/ncu",
        "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.bat",
        "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.exe",
    ]
    for c in candidates:
        if os.path.exists(c):
            return c

    raise FileNotFoundError(
        "ncu not found. Install NVIDIA Nsight Compute or add it to PATH.\n"
        "Download: https://developer.nvidia.com/nsight-compute"
    )


# ── Subprocess Runner ───────────────────────────────────────────

def run_under_ncu(
    ncu_binary: str,
    app_cmd: list[str],
    config: NCUConfig,
    output_file: str,
) -> subprocess.CompletedProcess:
    """Launch app_cmd under NCU profiling.

    Args:
        ncu_binary: path to ncu executable
        app_cmd: full command to run under NCU, e.g. [sys.executable, "bench.py", "--ncu-child"]
        config: NCU profiling parameters
        output_file: path for .ncu-rep (without extension)

    Returns:
        subprocess.CompletedProcess with captured stdout/stderr.
    """
    cmd = [ncu_binary] + config.to_cli_args() + ["-o", output_file, "-f"] + app_cmd
    return subprocess.run(cmd, capture_output=True, text=True, timeout=600)


# ── Metric Extraction ───────────────────────────────────────────

def parse_ncu_csv(csv_text: str) -> list[dict]:
    """Parse NCU CSV output into a list of per-kernel metric dicts.

    NCU CSV structure (with --page raw):
      Row 0: metric name headers
      Row 1: units
      Row 2+: data rows (one per kernel instance)

    Numeric values are parsed to floats with the metric name as key.
    """
    lines = [l for l in csv_text.strip().splitlines() if l.strip()]
    if len(lines) < 3:
        return []

    reader = csv.reader(lines)
    rows = list(reader)

    # rows[0] = header (metric names), rows[1] = units (ignored), rows[2:] = data
    headers = rows[0]
    data_rows = rows[2:]

    results = []
    for row in data_rows:
        entry = {}
        for i, h in enumerate(headers):
            if i < len(row):
                val = row[i]
                try:
                    entry[h] = float(val)
                except (ValueError, TypeError):
                    entry[h] = val
        results.append(entry)

    return results


def extract_metrics_from_report(
    ncu_binary: str,
    ncu_rep_file: str,
    metrics: list[str],
) -> list[dict]:
    """Extract specific metrics from an existing .ncu-rep file as parsed dicts."""
    cmd = [
        ncu_binary,
        "-i", ncu_rep_file,
        "--csv",
        "--page", "raw",
        "--metrics", ",".join(metrics),
    ]
    result = subprocess.check_output(cmd, text=True, timeout=120)
    return parse_ncu_csv(result)


# ── Bottleneck Classification ───────────────────────────────────

def classify_bottleneck(kernel_data: list[dict]) -> dict:
    """Aggregate metrics across kernels and classify the performance bottleneck.

    Uses the industry-standard heuristic:
      DRAM > 60% AND SM < 20%  → memory_bound
      SM   > 60% AND DRAM < 20% → compute_bound
      Both < 30%                → launch_occupancy_bound
      Otherwise                 → latency_bound

    Returns a dict with bottleneck label and aggregated percentages.
    """
    if not kernel_data:
        return {"bottleneck": "unknown", "sm_throughput_pct": 0.0,
                "dram_throughput_pct": 0.0, "occupancy_pct": 0.0}

    KEY_SM  = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    KEY_DRAM = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    KEY_OCC  = "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
    KEY_TIME = "gpu__time_duration.sum"

    # Weighted aggregation by kernel time
    total_time = sum(k.get(KEY_TIME, 0) for k in kernel_data)
    if total_time <= 0:
        total_time = 1.0

    weighted_sm = sum(
        k.get(KEY_SM, 0.0) * k.get(KEY_TIME, 0) for k in kernel_data
    ) / total_time
    weighted_dram = sum(
        k.get(KEY_DRAM, 0.0) * k.get(KEY_TIME, 0) for k in kernel_data
    ) / total_time
    weighted_occ = sum(
        k.get(KEY_OCC, 0.0) * k.get(KEY_TIME, 0) for k in kernel_data
    ) / total_time

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


# ── High-Level Profiling Entry Point ────────────────────────────

def profile_with_ncu(
    benchmark_script: str,
    op_args: list[str],
    ncu_config: NCUConfig,
    output_dir: str = "test/benchmark/ops/results/ncu",
) -> ProfileReport:
    """High-level entry point: run a benchmark script under NCU and return a report.

    Args:
        benchmark_script: path to e.g. "benchmark_linear.py"
        op_args: CLI args for the benchmark, e.g. ["--M", "512", "--dtype", "f16"]
        ncu_config: NCU profiling parameters
        output_dir: directory for .ncu-rep output

    Returns:
        ProfileReport with parsed metrics and bottleneck classification.
    """
    os.makedirs(output_dir, exist_ok=True)

    ncu_binary = find_ncu()
    op_name = os.path.basename(benchmark_script).replace("benchmark_", "").replace(".py", "")
    output_file = os.path.join(output_dir, f"{op_name}_{ncu_config.set}")

    # Build the child command
    app_cmd = [sys.executable, benchmark_script, "--ncu-child"] + op_args

    print(f"Profiling {op_name} under NCU ({ncu_config.set})...")
    result = run_under_ncu(ncu_binary, app_cmd, ncu_config, output_file)

    if result.returncode != 0:
        print(f"NCU stderr:\n{result.stderr}")
        raise RuntimeError(f"NCU profiling failed with code {result.returncode}")

    ncu_rep = output_file + ".ncu-rep"
    if not os.path.exists(ncu_rep):
        raise FileNotFoundError(f"Expected .ncu-rep not found: {ncu_rep}")

    # Extract and analyze
    kernel_data = extract_metrics_from_report(ncu_binary, ncu_rep, ncu_config.metrics)
    classif = classify_bottleneck(kernel_data)

    # Compute total time
    KEY_TIME = "gpu__time_duration.sum"
    total_time_ns = sum(k.get(KEY_TIME, 0) for k in kernel_data)

    return ProfileReport(
        operator=op_name,
        ncu_set=ncu_config.set,
        report_file=ncu_rep,
        kernel_count=len(kernel_data),
        total_time_us=total_time_ns / 1000.0,
        sm_throughput_pct=classif["sm_throughput_pct"],
        dram_throughput_pct=classif["dram_throughput_pct"],
        occupancy_pct=classif["occupancy_pct"],
        bottleneck=classif["bottleneck"],
        raw_metrics=kernel_data,
    )
```

- [ ] **Step 2: Verify NCU can be found**

```bash
cd test/benchmark/ops && python -c "
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname('.'), '..', '..')))
from ncu_profiler import find_ncu
try:
    print(f'NCU found at: {find_ncu()}')
except FileNotFoundError as e:
    print(f'Warning: {e}')
print('ncu_profiler imports OK')
"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/ncu_profiler.py
git commit -m "feat: add NCU profiler with subprocess double-launch and bottleneck classification"
```

---

### Task 4: benchmark_add.py — simple element-wise op (PyTorch ref)

**Files:**
- Create: `test/benchmark/ops/benchmark_add.py`

**Pattern for all simple-op benchmarks (add, argmax, embedding, rearrange).**
Each follows this template — subsequent tasks reference this pattern.

**Interfaces:**
- Consumes: `benchmark_harness` (run_benchmark, BenchmarkConfig), `ncu_profiler` (optional NCU)
- Produces: standalone CLI with `--M`, `--dtype`, `--ncu` flags

- [ ] **Step 1: Implement benchmark_add.py**

Write `test/benchmark/ops/benchmark_add.py`:

```python
#!/usr/bin/env python3
"""Benchmark: element-wise addition (add)."""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

# ── Operator parameters ─────────────────────────────────────────

OP_NAME = "add"
QWEN2_SHAPES = [
    # (rows, cols) — element-wise so N/K concept doesn't apply,
    # but we use model-relevant 2D shapes for realistic bandwidth measurement
    {"shape": (1, 3584)},       # decode, single token
    {"shape": (16, 3584)},      # small decode batch
    {"shape": (64, 3584)},      # medium decode
    {"shape": (128, 3584)},     # large decode
    {"shape": (512, 3584)},     # small prefill
    {"shape": (2048, 3584)},    # large prefill
]


# ── Reference implementation ─────────────────────────────────────

def torch_add(buffers: dict) -> None:
    """Reference: PyTorch CUDA element-wise add."""
    torch.add(buffers["a_torch"], buffers["b_torch"], out=buffers["out_torch"])


def llaisys_add(buffers: dict) -> None:
    llaisys.Ops.add(buffers["out"], buffers["a"], buffers["b"])


# ── Setup ───────────────────────────────────────────────────────

def setup_add(shape: tuple, dtype_name: str) -> dict:
    """Allocate input/output tensors for one benchmark iteration."""
    a_torch, a = random_tensor(shape, dtype_name, "nvidia")
    b_torch, b = random_tensor(shape, dtype_name, "nvidia")
    out_torch, out = random_tensor(shape, dtype_name, "nvidia")

    return {
        "a_torch": a_torch, "a": a,
        "b_torch": b_torch, "b": b,
        "out_torch": out_torch, "out": out,
    }


# ── Main ────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
    parser.add_argument("--ncu-child", action="store_true",
                        help=argparse.SUPPRESS)  # internal flag for NCU subprocess
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    device_name = "nvidia"
    dtype_name = args.dtype

    results = []
    for shape_info in QWEN2_SHAPES:
        shape = shape_info["shape"]
        N = shape[0] * shape[1]
        esize = elem_size(dtype_name)
        total_bytes = N * esize

        config = BenchmarkConfig(
            op_name=OP_NAME,
            shape={"rows": shape[0], "cols": shape[1]},
            dtype_name=dtype_name,
            warmup=args.warmup,
            repeat=args.repeat,
            total_bytes_read=2 * total_bytes,   # a + b
            total_bytes_write=total_bytes,       # out
            total_flops=0,                       # no FLOPS for element-wise
        )

        def setup_fn(s=shape, d=dtype_name):
            return setup_add(s, d)

        def llsys_fn(buf):
            llaisys_add(buf)

        def ref_fn(buf):
            torch_add(buf)

        result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
        result.baseline_name = "torch"
        results.append(result)

        # Quick correctness check on first shape
        if shape == QWEN2_SHAPES[0]["shape"]:
            buf = setup_add(shape, dtype_name)
            torch_add(buf)
            torch_ref = buf["out_torch"].clone()
            llaisys_add(buf)
            ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
            print(f"  Correctness: {'PASS' if ok else 'FAIL'}")
            assert ok, "Add correctness check failed!"

        print(format_summary(result))

    # ── NCU profiling (if requested) ──
    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(
            set=set_map[args.ncu],
            kernel_name=f".*{OP_NAME}.*",
        )
        for shape_info in QWEN2_SHAPES[:1]:  # profile first shape only
            shape = shape_info["shape"]
            report = profile_with_ncu(
                __file__,
                ["--dtype", dtype_name, "--warmup", "5", "--repeat", "1"],
                ncu_config,
                os.path.join(args.output, "ncu"),
            )
            # Merge NCU data into first result
            results[0].ncu_set = report.ncu_set
            results[0].ncu_report_file = report.report_file
            results[0].ncu_bottleneck = report.bottleneck
            results[0].ncu_sm_throughput_pct = report.sm_throughput_pct
            results[0].ncu_dram_throughput_pct = report.dram_throughput_pct
            results[0].ncu_occupancy_pct = report.occupancy_pct

    # ── NCU child mode: just run once and exit ──
    if args.ncu_child:
        buf = setup_add(QWEN2_SHAPES[0]["shape"], dtype_name)
        llaisys_add(buf)
        torch.cuda.synchronize()
        return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark_add.py**

```bash
python test/benchmark/ops/benchmark_add.py --dtype f16 --warmup 5 --repeat 20
```

Expected: output shows add benchmark results with min/median/mean/stddev, torch baseline comparison.

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/benchmark_add.py
git commit -m "feat: add benchmark_add with CUDA event timing and torch baseline"
```

---

### Task 5: benchmark_argmax.py

**Files:**
- Create: `test/benchmark/ops/benchmark_argmax.py`

Follows the same pattern as Task 4 (benchmark_add.py).
Key differences:

```python
OP_NAME = "argmax"

QWEN2_SHAPES = [
    {"shape": (1, 3584)},       # decode
    {"shape": (64, 3584)},      # medium batch
    {"shape": (128, 3584)},     # large decode
    {"shape": (512, 3584)},     # prefill
    {"shape": (2048, 3584)},    # large prefill
    {"shape": (2048, 18944)},   # intermediate_size dim
]

def setup_argmax(shape: tuple, dtype_name: str) -> dict:
    vals_torch, vals = random_tensor(shape, dtype_name, "nvidia")
    max_val_torch, max_val = random_tensor((shape[0],), dtype_name, "nvidia")
    max_idx_torch, max_idx = random_int_tensor((shape[0],), "nvidia", "i64")
    return {
        "vals_torch": vals_torch, "vals": vals,
        "max_val_torch": max_val_torch, "max_val": max_val,
        "max_idx_torch": max_idx_torch, "max_idx": max_idx,
    }

def torch_argmax(buf):
    torch.argmax(buf["vals_torch"], dim=-1, out=buf["max_idx_torch"])
    # also compute max values for completeness
    torch.max(buf["vals_torch"], dim=-1, out=(buf["max_val_torch"],))

def llaisys_argmax(buf):
    llaisys.Ops.argmax(buf["max_idx"], buf["max_val"], buf["vals"])
```

All other patterns (main(), NCU handling, correctness check) identical to Task 4.

- [ ] **Step 1: Implement and test benchmark_argmax.py**

- [ ] **Step 2: Commit**

---

### Task 6: benchmark_embedding.py

**Files:**
- Create: `test/benchmark/ops/benchmark_embedding.py`

Follows Task 4 pattern. Key differences:

```python
OP_NAME = "embedding"

# vocab_size=151936, hidden_size=3584 (Qwen2-7B)
QWEN2_SHAPES = [
    {"seqlen": 1, "vocab": 151936, "dim": 3584},      # decode
    {"seqlen": 16, "vocab": 151936, "dim": 3584},      # small batch
    {"seqlen": 64, "vocab": 151936, "dim": 3584},
    {"seqlen": 128, "vocab": 151936, "dim": 3584},
    {"seqlen": 512, "vocab": 151936, "dim": 3584},     # prefill
    {"seqlen": 2048, "vocab": 151936, "dim": 3584},    # large prefill
]

def setup_embedding(seqlen, vocab, dim, dtype_name):
    idx_torch, idx = random_int_tensor((seqlen,), "nvidia", "i64", low=0, high=vocab)
    w_torch, w = random_tensor((vocab, dim), dtype_name, "nvidia")
    out_torch, out = random_tensor((seqlen, dim), dtype_name, "nvidia")
    return {"idx_torch": idx_torch, "idx": idx, "w_torch": w_torch, "w": w,
            "out_torch": out_torch, "out": out}

def torch_embedding(buf):
    torch.nn.functional.embedding(buf["idx_torch"], buf["w_torch"], out=buf["out_torch"])

def llaisys_embedding(buf):
    llaisys.Ops.embedding(buf["out"], buf["idx"], buf["w"])
```

- [ ] **Step 1: Implement and test benchmark_embedding.py**

- [ ] **Step 2: Commit**

---

### Task 7: benchmark_rearrange.py

**Files:**
- Create: `test/benchmark/ops/benchmark_rearrange.py`

Follows Task 4 pattern. Key differences:

```python
OP_NAME = "rearrange"

# M=seqlen, N=nh*dh=3584 (Q proj output before/after reshape)
QWEN2_SHAPES = [
    {"shape": (1, 3584)},       # decode
    {"shape": (16, 3584)},
    {"shape": (64, 3584)},
    {"shape": (128, 3584)},
    {"shape": (512, 3584)},     # prefill
    {"shape": (2048, 3584)},    # large prefill
]

def setup_rearrange(shape, dtype_name):
    a_torch, a = random_tensor(shape, dtype_name, "nvidia")
    # Target: same shape but permuted strides (simulate reshape)
    b_torch, b = random_tensor(shape, dtype_name, "nvidia")
    return {"a_torch": a_torch, "a": a, "b_torch": b_torch, "b": b}

def torch_rearrange(buf):
    # rearrange is effectively contiguous() when strides differ
    buf["b_torch"].copy_(buf["a_torch"])

def llaisys_rearrange(buf):
    llaisys.Ops.rearrange(buf["b"], buf["a"])
```

- [ ] **Step 1: Implement and test benchmark_rearrange.py**

- [ ] **Step 2: Commit**

---

### Task 8: benchmark_linear.py — cuBLAS baseline, multiple (N,K) pairs

**Files:**
- Create: `test/benchmark/ops/benchmark_linear.py`

**Interfaces:**
- Consumes: benchmark_harness, ncu_profiler
- Produces: benchmark_linear CLI

- [ ] **Step 1: Implement benchmark_linear.py**

Write `test/benchmark/ops/benchmark_linear.py`:

```python
#!/usr/bin/env python3
"""Benchmark: linear (matrix multiplication) — cuBLAS is the gold-standard baseline.

Covers all Qwen2 linear layer variants:
  - Q/K/V projection:  in [M, 3584] × weight [3584, 3584]
  - Gate/Up projection: in [M, 3584] × weight [18944, 3584]
  - Down projection:    in [M, 18944] × weight [3584, 18944]
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
import llaisys
from test_utils import random_tensor, check_equal

from benchmark_harness import (
    BenchmarkConfig, BenchmarkResult, run_benchmark, format_summary, save_results, elem_size
)

OP_NAME = "linear"

# (M, N, K) = (seq_tokens, out_features, in_features)
# Each variant: 6 M values
QWEN2_VARIANTS = {
    "q_proj":    {"N": 3584,  "K": 3584,  "label": "Q/K/V/O proj"},
    "gate_proj": {"N": 18944, "K": 3584,  "label": "Gate/Up proj"},
    "down_proj": {"N": 3584,  "K": 18944, "label": "Down proj"},
}
M_VALUES = [1, 16, 64, 128, 512, 2048]


def setup_linear(M: int, N: int, K: int, dtype_name: str, use_bias: bool = True) -> dict:
    """Allocate in/weight/out tensors for one linear benchmark iteration."""
    in_t, inp = random_tensor((M, K), dtype_name, "nvidia", scale=0.1)
    w_t, w = random_tensor((N, K), dtype_name, "nvidia", scale=0.01)
    out_t, out = random_tensor((M, N), dtype_name, "nvidia")

    bias, bias_t = None, None
    if use_bias:
        bias_t, bias = random_tensor((N,), dtype_name, "nvidia")

    return {
        "in_t": in_t, "in": inp,
        "w_t": w_t, "w": w,
        "out_t": out_t, "out": out,
        "bias_t": bias_t, "bias": bias,
        "use_bias": use_bias,
    }


def torch_linear(buf: dict) -> None:
    bias = buf["bias_t"] if buf["use_bias"] else None
    torch.nn.functional.linear(buf["in_t"], buf["w_t"], bias, out=buf["out_t"])


def llaisys_linear(buf: dict) -> None:
    llaisys.Ops.linear(buf["out"], buf["in"], buf["w"], buf["bias"])


def main():
    import argparse
    parser = argparse.ArgumentParser(description=f"Benchmark {OP_NAME} operator")
    parser.add_argument("--variant", default="all", choices=["all", "q_proj", "gate_proj", "down_proj"])
    parser.add_argument("--M", type=int, default=0, help="Override M; 0 = sweep all M values")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
    parser.add_argument("--ncu-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--output", default="test/benchmark/ops/results")
    args = parser.parse_args()

    device_name = "nvidia"
    dtype_name = args.dtype
    esize = elem_size(dtype_name)
    use_bias = True  # Qwen2 uses bias

    variants = list(QWEN2_VARIANTS.items()) if args.variant == "all" else \
               [(args.variant, QWEN2_VARIANTS[args.variant])]

    M_list = [args.M] if args.M > 0 else M_VALUES

    results = []
    for var_name, var_info in variants:
        N, K = var_info["N"], var_info["K"]
        for M in M_list:
            # GEMM FLOPs: 2 * M * N * K
            flops = 2 * M * N * K
            total_bytes = (M * K + N * K + M * N) * esize  # in + weight + out

            config = BenchmarkConfig(
                op_name=f"{OP_NAME}/{var_name}",
                shape={"M": M, "N": N, "K": K},
                dtype_name=dtype_name,
                warmup=args.warmup,
                repeat=args.repeat,
                total_bytes_read=(M*K+N*K)*esize  + (N*esize if use_bias else 0),
                total_bytes_write=M*N*esize,
                total_flops=flops,
            )

            def setup_fn(m=M, n=N, k=K, d=dtype_name):
                return setup_linear(m, n, k, d, use_bias)

            def llsys_fn(buf): llaisys_linear(buf)
            def ref_fn(buf): torch_linear(buf)

            result = run_benchmark(config, setup_fn, llsys_fn, ref_fn)
            result.baseline_name = "cuBLAS"
            results.append(result)

            # Correctness check on first config
            if M == M_VALUES[0] and var_name == variants[0][0]:
                buf = setup_linear(M, N, K, dtype_name, use_bias)
                torch_linear(buf)
                torch_ref = buf["out_t"].clone()
                torch.cuda.synchronize()
                llaisys_linear(buf)
                torch.cuda.synchronize()
                ok = check_equal(buf["out"], torch_ref, atol=1e-2, rtol=1e-2)
                print(f"  Correctness ({var_name} M={M}): {'PASS' if ok else 'FAIL'}")
                assert ok, f"Linear correctness check failed for {var_name}!"

            print(format_summary(result))

    # ── NCU profiling ──
    if args.ncu and not args.ncu_child:
        from ncu_profiler import NCUConfig, profile_with_ncu
        set_map = {"quick": "default", "detailed": "detailed", "full": "full"}
        ncu_config = NCUConfig(set=set_map[args.ncu], kernel_name=".*linear.*")
        # Profile mid-size config
        M, N, K = 128, 3584, 3584
        report = profile_with_ncu(
            __file__,
            ["--variant", "q_proj", "--M", str(M), "--dtype", dtype_name,
             "--warmup", "5", "--repeat", "1"],
            ncu_config,
            os.path.join(args.output, "ncu"),
        )
        results[0].ncu_set = report.ncu_set
        results[0].ncu_report_file = report.report_file
        results[0].ncu_bottleneck = report.bottleneck
        results[0].ncu_sm_throughput_pct = report.sm_throughput_pct
        results[0].ncu_dram_throughput_pct = report.dram_throughput_pct
        results[0].ncu_occupancy_pct = report.occupancy_pct

    if args.ncu_child:
        variant = variants[0]
        M = 128
        N, K = variant[1]["N"], variant[1]["K"]
        buf = setup_linear(M, N, K, dtype_name, use_bias)
        llaisys_linear(buf)
        torch.cuda.synchronize()
        return

    save_results(results, args.output)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run benchmark_linear.py**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --warmup 5 --repeat 20 --variant q_proj --M 512
```

Expected: LLAISYS vs cuBLAS timing, TFLOPS calculation, correctness PASS.

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/benchmark_linear.py
git commit -m "feat: add benchmark_linear with cuBLAS baseline and Qwen2 shape variants"
```

---

### Task 9: benchmark_rms_norm.py — Unsloth Triton baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_rms_norm.py`

Follows Task 8 pattern. Key differences — the reference is an inlined Triton RMSNorm kernel (simplified Unsloth-style) rather than PyTorch:

```python
OP_NAME = "rms_norm"

QWEN2_SHAPES = [
    {"shape": (1, 3584)},       # decode
    {"shape": (16, 3584)},
    {"shape": (64, 3584)},
    {"shape": (128, 3584)},
    {"shape": (512, 3584)},     # prefill
    {"shape": (2048, 3584)},    # large prefill
]

def setup_rms_norm(shape, dtype_name):
    inp_t, inp = random_tensor(shape, dtype_name, "nvidia")
    w_t, w = random_tensor((shape[1],), dtype_name, "nvidia")
    out_t, out = random_tensor(shape, dtype_name, "nvidia")
    return {"in_t": inp_t, "in": inp, "w_t": w_t, "w": w, "out_t": out_t, "out": out}

# Reference: PyTorch RMSNorm (torch doesn't have a fused rms_norm, so we use
# the standard decomposed implementation as the "baseline" — it calls cuBLAS
# and elementwise kernels that represent the best available fused approach)
def torch_rms_norm(buf):
    x, w = buf["in_t"], buf["w_t"]
    # x * w / sqrt(mean(x^2) + eps)
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-6)
    buf["out_t"].copy_(x * w / rms)

def llaisys_rms_norm(buf):
    llaisys.Ops.rms_norm(buf["out"], buf["in"], buf["w"], 1e-6)
```

Baseline is `"torch_rms_norm"` — noted in result as decomposed PyTorch RMSNorm.

- [ ] **Step 1: Implement and test benchmark_rms_norm.py**

- [ ] **Step 2: Commit**

---

### Task 10: benchmark_rope.py — Unsloth fused QK-RoPE baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_rope.py`

```python
OP_NAME = "rope"

# (seqlen, nh, dh) = (M, 28, 128) — Qwen2-7B attention head dims
# Also cover nkvh=4 for K projection
QWEN2_SHAPES = [
    {"seqlen": 1, "nhead": 28, "hd": 128},       # decode (Q)
    {"seqlen": 1, "nhead": 4, "hd": 128},        # decode (K, GQA)
    {"seqlen": 16, "nhead": 28, "hd": 128},
    {"seqlen": 128, "nhead": 28, "hd": 128},
    {"seqlen": 512, "nhead": 28, "hd": 128},      # prefill
    {"seqlen": 2048, "nhead": 28, "hd": 128},     # large prefill
]
THETA = 10000.0

def setup_rope(seqlen, nhead, hd, dtype_name):
    inp_t, inp = random_tensor((seqlen, nhead, hd), dtype_name, "nvidia")
    pos_t, pos = random_int_tensor((seqlen,), "nvidia", "i64", low=0, high=2048)
    out_t, out = random_tensor((seqlen, nhead, hd), dtype_name, "nvidia")
    return {"in_t": inp_t, "in": inp, "pos_t": pos_t, "pos": pos,
            "out_t": out_t, "out": out}

# PyTorch reference: explicit cos/sin computation (faithful to algorithm,
# not optimized — RoPE has no single fused PyTorch op)
def torch_rope(buf):
    """Reference RoPE implementation (in-place on the output tensor)."""
    inp, pos = buf["in_t"], buf["pos_t"]
    seqlen, nhead, hd = inp.shape
    d2 = hd // 2
    # Compute frequencies
    theta = 10000.0
    j = torch.arange(0, d2, device=inp.device, dtype=inp.dtype)
    freq = 1.0 / (theta ** (2.0 * j / hd))
    # [seqlen, d/2]
    phi = pos.float().unsqueeze(1) / freq.unsqueeze(0)
    cos, sin = torch.cos(phi), torch.sin(phi)
    cos = cos.unsqueeze(1).expand(-1, nhead, -1)
    sin = sin.unsqueeze(1).expand(-1, nhead, -1)
    # Interleave: [a1,b1,a2,b2,...] → rotate adjacent pairs
    a = inp[..., 0::2]
    b = inp[..., 1::2]
    buf["out_t"][..., 0::2] = a * cos - b * sin
    buf["out_t"][..., 1::2] = b * cos + a * sin

def llaisys_rope(buf):
    llaisys.Ops.rope(buf["out"], buf["in"], buf["pos"], THETA)
```

- [ ] **Step 1: Implement and test benchmark_rope.py**

- [ ] **Step 2: Commit**

---

### Task 11: benchmark_swiglu.py — Unsloth fused SwiGLU baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_swiglu.py`

```python
OP_NAME = "swiglu"

# (M, di) where di=18944 (Qwen2 intermediate_size)
QWEN2_SHAPES = [
    {"shape": (1, 18944)},       # decode
    {"shape": (16, 18944)},
    {"shape": (64, 18944)},
    {"shape": (128, 18944)},
    {"shape": (512, 18944)},     # prefill
    {"shape": (2048, 18944)},    # large prefill
]

def setup_swiglu(shape, dtype_name):
    gate_t, gate = random_tensor(shape, dtype_name, "nvidia")
    up_t, up = random_tensor(shape, dtype_name, "nvidia")
    out_t, out = random_tensor(shape, dtype_name, "nvidia")
    return {"gate_t": gate_t, "gate": gate, "up_t": up_t, "up": up,
            "out_t": out_t, "out": out}

# PyTorch: silu(gate) * up (two separate kernel launches, no fused kernel)
def torch_swiglu(buf):
    buf["out_t"].copy_(torch.nn.functional.silu(buf["gate_t"]) * buf["up_t"])

def llaisys_swiglu(buf):
    llaisys.Ops.swiglu(buf["out"], buf["gate"], buf["up"])
```

- [ ] **Step 1: Implement and test benchmark_swiglu.py**

- [ ] **Step 2: Commit**

---

### Task 12: benchmark_self_attention.py — Flash-Attention v2 baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_self_attention.py`

```python
OP_NAME = "self_attention"

# (seqlen, nhead/nkvh, hd) with Qwen2 dims
QWEN2_SHAPES = [
    # seqlen, total_len, nh, nkvh, hd
    {"seqlen": 1, "total_len": 64, "nh": 28, "nkvh": 4, "hd": 128},    # decode
    {"seqlen": 1, "total_len": 512, "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 16, "total_len": 128, "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 64, "total_len": 128, "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 128, "total_len": 256, "nh": 28, "nkvh": 4, "hd": 128},  # small prefill
    {"seqlen": 512, "total_len": 1024, "nh": 28, "nkvh": 4, "hd": 128}, # large prefill
]
SCALE = 1.0 / (128 ** 0.5)

def setup_self_attn(seqlen, total_len, nh, nkvh, hd, dtype_name):
    q_t, q = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    k_t, k = random_tensor((total_len, nkvh, hd), dtype_name, "nvidia")
    v_t, v = random_tensor((total_len, nkvh, hd), dtype_name, "nvidia")
    out_t, out = random_tensor((seqlen, nh, hd), dtype_name, "nvidia")
    return {"q_t": q_t, "q": q, "k_t": k_t, "k": k, "v_t": v_t, "v": v,
            "out_t": out_t, "out": out}

# PyTorch scaled_dot_product_attention uses Flash-Attention v2 under the hood
def torch_self_attn(buf):
    q, k, v = buf["q_t"], buf["k_t"], buf["v_t"]
    # GQA: repeat KV heads to match Q heads
    nh, nkvh = q.shape[1], k.shape[1]
    if nh != nkvh:
        repeat = nh // nkvh
        k = k.repeat_interleave(repeat, dim=1)
        v = v.repeat_interleave(repeat, dim=1)
    buf["out_t"].copy_(torch.nn.functional.scaled_dot_product_attention(
        q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1),
        scale=SCALE, is_causal=True,
    ).transpose(0, 1))

def llaisys_self_attn(buf):
    llaisys.Ops.self_attention(buf["out"], buf["q"], buf["k"], buf["v"], SCALE)
```

- [ ] **Step 1: Implement and test benchmark_self_attention.py**

- [ ] **Step 2: Commit**

---

### Task 13: benchmark_topk.py — PyTorch reference

**Files:**
- Create: `test/benchmark/ops/benchmark_topk.py`

```python
OP_NAME = "top_k"
K = 10  # matches TOP_K in src/config.hpp

# 1D and 2D (batched) inputs
QWEN2_SHAPES = [
    {"shape": (3584,), "k": K},           # single vector, hidden_size
    {"shape": (18944,), "k": K},          # intermediate_size
    {"shape": (1, 32000), "k": K},        # batched, vocab-sized
    {"shape": (16, 32000), "k": K},       # batched×16
    {"shape": (64, 32000), "k": K},
    {"shape": (128, 32000), "k": K},
]

def setup_topk(shape, k, dtype_name):
    vals_t, vals = random_tensor(shape, dtype_name, "nvidia")
    batch = 1 if len(shape) == 1 else shape[0]
    out_val_t, out_val = random_tensor((batch, k), dtype_name, "nvidia")
    out_idx_t, out_idx = random_int_tensor((batch, k), "nvidia", "i64")
    return {"vals_t": vals_t, "vals": vals,
            "out_val_t": out_val_t, "out_val": out_val,
            "out_idx_t": out_idx_t, "out_idx": out_idx, "k": k}

def torch_topk(buf):
    k = buf["k"]
    vals = buf["vals_t"]
    topk_vals, topk_idx = torch.topk(vals, k, dim=-1)
    buf["out_val_t"].copy_(topk_vals)
    buf["out_idx_t"].copy_(topk_idx)

def llaisys_topk(buf):
    llaisys.Ops.topk(buf["out_idx"], buf["out_val"], buf["vals"], buf["k"])
```

- [ ] **Step 1: Implement and test benchmark_topk.py**

- [ ] **Step 2: Commit**

---

### Task 14: benchmark_paged_attention.py — no baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_paged_attention.py`

**Key difference:** `ref_fn=None` — no baseline comparison.

```python
OP_NAME = "paged_attention"
KV_CACHE_TOKEN_NUM = 64  # from src/config.hpp

QWEN2_SHAPES = [
    # seqlen, token_num, block_num, block_ids, nh, nkvh, hd
    {"seqlen": 1, "tk": 64, "bn": 8, "bids": [0, 1, 2, 3], "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 16, "tk": 64, "bn": 8, "bids": [0, 1, 2, 3, 4, 5], "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 128, "tk": 64, "bn": 8, "bids": [0, 1, 2, 3, 4, 5, 6, 7], "nh": 28, "nkvh": 4, "hd": 128},
    {"seqlen": 512, "tk": 64, "bn": 16, "bids": list(range(16)), "nh": 28, "nkvh": 4, "hd": 128},
]

def setup_paged_attn(seqlen, token_num, block_num, block_ids, nh, nkvh, hd, dtype_name):
    """Allocate Q, KV-cache, block-table tensors."""
    # ... (adapted from test/ops/paged_attention.py::test_op_paged_attention_batched)
    # Returns dict with q, k_cache, v_cache, attn_val, block_ids, cut_idx, tot_len
    pass

def llaisys_paged_attn(buf):
    llaisys.Ops.paged_attention(
        buf["attn_val"], buf["q"], buf["k_cache"], buf["v_cache"],
        buf["block_ids"], buf["cut_idx"], buf["tot_len"],
        buf["max_seq_len"], 1.0 / (buf["hd"] ** 0.5),
    )
```

The setup function builds the block table and flat KV cache tensors following the pattern in `test/ops/paged_attention.py:49-213` (building `block_ids_ll`, `cut_idx_ll`, `totlen_ll`).

Benchmark reports only absolute metrics (latency, bandwidth, TFLOPS, occupancy from NCU).

- [ ] **Step 1: Implement and test benchmark_paged_attention.py**

- [ ] **Step 2: Commit**

---

### Task 15: benchmark_kv_cache_move.py — no baseline

**Files:**
- Create: `test/benchmark/ops/benchmark_kv_cache_move.py`

```python
OP_NAME = "kv_cache_move"
KV_CACHE_TOKEN_NUM = 64

QWEN2_SHAPES = [
    # batch, max_seq_len, token_num, max_block_num, nkvh, dh
    {"batch": 1, "max_seq_len": 64, "tk": 64, "max_blocks": 8, "nkvh": 4, "dh": 128},
    {"batch": 1, "max_seq_len": 512, "tk": 64, "max_blocks": 8, "nkvh": 4, "dh": 128},
    {"batch": 4, "max_seq_len": 64, "tk": 64, "max_blocks": 8, "nkvh": 4, "dh": 128},
    {"batch": 8, "max_seq_len": 128, "tk": 64, "max_blocks": 16, "nkvh": 4, "dh": 128},
]

def setup_kv_cache_move(batch, max_seq_len, token_num, max_block_num, nkvh, dh, dtype_name):
    """Allocate input (flat), output (blocked K/V cache), block table, cut_idx, pos_ids."""
    # out: [batch, token_num, nkvh, dh]  → blocked cache format
    # in:  [batch, max_seq_len, nkvh, dh]  → flat K/V tensor
    pass

def llaisys_kv_cache_move(buf):
    llaisys.Ops.kv_cache_move(
        buf["out"], buf["in"], buf["block_ids"],
        buf["cut_idx"], buf["pos_ids"], buf["max_seq_len"],
    )
```

- [ ] **Step 1: Implement and test benchmark_kv_cache_move.py**

- [ ] **Step 2: Commit**

---

### Task 16: run_all_benchmarks.py — batch runner

**Files:**
- Create: `test/benchmark/ops/run_all_benchmarks.py`

**Interfaces:**
- Consumes: all benchmark_*.py scripts (runs them as subprocesses)
- Produces: aggregated JSON at results/latest.json

- [ ] **Step 1: Implement run_all_benchmarks.py**

```python
#!/usr/bin/env python3
"""Batch runner: execute all operator benchmarks and aggregate results."""

import json
import os
import subprocess
import sys
import time
from datetime import datetime

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))

OPERATORS = [
    "add", "argmax", "embedding", "rearrange",
    "linear", "rms_norm", "rope", "swiglu",
    "self_attention", "topk",
    "paged_attention", "kv_cache_move",
]

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all operator benchmarks")
    parser.add_argument("--dtype", default="f16", choices=["f16", "bf16", "f32"])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--op", default="all", help="Comma-separated list or 'all'")
    parser.add_argument("--ncu", nargs="?", const="detailed",
                        choices=["quick", "detailed", "full"])
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

        cmd = [
            sys.executable, script,
            "--dtype", args.dtype,
            "--warmup", str(args.warmup),
            "--repeat", str(args.repeat),
            "--output", args.output,
        ]
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
```

- [ ] **Step 2: Test batch runner**

```bash
python test/benchmark/ops/run_all_benchmarks.py --op add --dtype f16 --warmup 5 --repeat 20
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/run_all_benchmarks.py
git commit -m "feat: add batch benchmark runner for all operators"
```

---

### Task 17: ncu_profile.sh — shell convenience wrapper

**Files:**
- Create: `test/benchmark/ops/ncu_profile.sh`

- [ ] **Step 1: Implement ncu_profile.sh**

```bash
#!/bin/bash
# NCU profiling convenience wrapper for operator benchmarks.
# Usage: bash ncu_profile.sh --op linear --set detailed --dtype f16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/ncu"
mkdir -p "${OUT_DIR}"

NCU=$(which ncu 2>/dev/null || echo "")
if [ -z "$NCU" ]; then
    # Try WSL Windows-side NCU
    NCU="/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.bat"
fi

OP="${OP:-linear}"
NCU_SET="${NCU_SET:-detailed}"
DTYPE="${DTYPE:-f16}"

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --op) OP="$2"; shift 2 ;;
        --set) NCU_SET="$2"; shift 2 ;;
        --dtype) DTYPE="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

BENCH_SCRIPT="${SCRIPT_DIR}/benchmark_${OP}.py"

if [ ! -f "$BENCH_SCRIPT" ]; then
    echo "ERROR: Benchmark script not found: $BENCH_SCRIPT"
    exit 1
fi

echo "=== NCU Profiling: ${OP} (${NCU_SET}, ${DTYPE}) ==="

"${NCU}" \
    --set "${NCU_SET}" \
    --clock-control base \
    --launch-skip 10 \
    --launch-count 1 \
    --target-processes all \
    -o "${OUT_DIR}/${OP}_${NCU_SET}" \
    -f \
    python "${BENCH_SCRIPT}" \
        --ncu-child \
        --dtype "${DTYPE}" \
        --warmup 5 --repeat 1

echo ""
echo "=== Extracting metrics from .ncu-rep ==="
"${NCU}" -i "${OUT_DIR}/${OP}_${NCU_SET}.ncu-rep" \
    --csv --page raw \
    --metrics gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed

echo ""
echo "Report saved to: ${OUT_DIR}/${OP}_${NCU_SET}.ncu-rep"
echo "Open with: ncu-ui ${OUT_DIR}/${OP}_${NCU_SET}.ncu-rep"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x test/benchmark/ops/ncu_profile.sh
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/ops/ncu_profile.sh
git commit -m "feat: add NCU profiling shell wrapper"
```

---

## Verification

After all tasks are complete, run the full verification:

- [ ] **V1: Single operator benchmark**

```bash
python test/benchmark/ops/benchmark_add.py --dtype f16 --warmup 5 --repeat 20
```
Expected: correctness PASS, latency/bandwidth values printed, torch baseline comparison.

- [ ] **V2: Linear benchmark with all variants**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --warmup 5 --repeat 20
```
Expected: results for Q/K/V/O + Gate/Up + Down proj across all M values.

- [ ] **V3: Batch runner**

```bash
python test/benchmark/ops/run_all_benchmarks.py --op add,linear --dtype f16 --warmup 5 --repeat 20
```
Expected: both operators run, aggregated JSON at results/latest.json.

- [ ] **V4: NCU profiling (if NCU available)**

```bash
python test/benchmark/ops/benchmark_linear.py --dtype f16 --ncu quick --M 128 --variant q_proj --warmup 5 --repeat 1
```
Expected: .ncu-rep file generated, bottleneck classification printed.

- [ ] **V5: NCU shell wrapper**

```bash
bash test/benchmark/ops/ncu_profile.sh --op linear --set default
```
Expected: same as V4 but via shell wrapper.

- [ ] **V6: JSON output**

```bash
python -c "
import json
with open('test/benchmark/ops/results/latest.json') as f:
    data = json.load(f)
print(f'Environment: {data[\"environment\"]}')
print(f'Results: {len(data[\"results\"])} entries')
for r in data['results']:
    print(f'  {r[\"operator\"]:20s} {r[\"dtype\"]:5s} {r[\"shape\"]} → {r[\"latency_us\"]:.1f} us')
"
```
Expected: structured JSON with environment metadata and result entries.

