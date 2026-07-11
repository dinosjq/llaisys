"""
Shared CUDA operator benchmarking framework.

Provides GPU-side timing via torch.cuda.Event, buffer rotation to defeat
L2 cache, and result collection with statistics and JSON output.

CUDA (NVIDIA) only.  Min is the primary metric (CUTLASS convention).
"""

import dataclasses
import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional

import torch


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def elem_size(dtype_name: str) -> int:
    """Map dtype name to element size in bytes."""
    mapping = {
        "f16": 2,
        "bf16": 2,
        "f32": 4,
        "f64": 8,
        "i32": 4,
        "i64": 8,
        "u32": 4,
        "u64": 8,
        "bool": 1,
    }
    return mapping.get(dtype_name, 4)


# ---------------------------------------------------------------------------
# GPU-side timer
# ---------------------------------------------------------------------------

class CUDATimer:
    """GPU-side timer using torch.cuda.Event with timing enabled."""

    def __init__(self) -> None:
        self._start: Optional[torch.cuda.Event] = None
        self._end: Optional[torch.cuda.Event] = None

    def record_start(self) -> None:
        """Record a start event on the current CUDA stream."""
        self._start = torch.cuda.Event(enable_timing=True)
        self._start.record()

    def record_end(self) -> None:
        """Record an end event on the current CUDA stream."""
        self._end = torch.cuda.Event(enable_timing=True)
        self._end.record()

    def elapsed_ms(self) -> float:
        """Synchronise on the end event and return elapsed time in ms."""
        if self._start is None or self._end is None:
            raise RuntimeError("CUDATimer: record_start() and record_end() must "
                               "be called before elapsed_ms()")
        self._end.synchronize()
        return self._start.elapsed_time(self._end)  # pyright: ignore[reportArgumentType]


# ---------------------------------------------------------------------------
# buffer rotation
# ---------------------------------------------------------------------------

class BufferPool:
    """Rotates through N buffer sets to defeat L2 cache warm-up effects."""

    def __init__(self, n_sets: int, alloc_fn: Callable[[], Any]) -> None:
        if n_sets < 1:
            raise ValueError("BufferPool requires n_sets >= 1")
        self._sets = [alloc_fn() for _ in range(n_sets)]
        self._index = 0

    def next(self) -> Any:
        """Return the current buffer set and advance the rotation."""
        item = self._sets[self._index]
        self._index = (self._index + 1) % len(self._sets)
        return item

    def __len__(self) -> int:
        return len(self._sets)


# ---------------------------------------------------------------------------
# benchmark configuration
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkConfig:
    """Configuration for a single operator benchmark run."""

    op_name: str
    shape: dict
    dtype_name: str
    warmup: int = 10
    repeat: int = 100
    total_bytes_read: int = 0
    total_bytes_write: int = 0
    total_flops: int = 0


# ---------------------------------------------------------------------------
# benchmark result
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Latency and throughput statistics for one benchmark configuration."""

    operator: str
    dtype: str
    shape: dict
    latency_us: float                # primary metric: min latency in microseconds
    latency_stats: dict              # {min, median, mean, stddev, p99}
    bandwidth_GBs: float
    TFLOPS: float

    # -- baseline comparison (populated when ref_fn is provided) --
    baseline_name: str = ""
    baseline_latency_us: float = 0.0
    baseline_TFLOPS: float = 0.0
    speedup: float = 0.0

    # -- NCU profiling metadata (populated by external NCU tooling) --
    ncu_set: str = ""
    ncu_report_file: str = ""
    ncu_bottleneck: str = ""
    ncu_sm_throughput_pct: float = 0.0
    ncu_dram_throughput_pct: float = 0.0
    ncu_occupancy_pct: float = 0.0

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @staticmethod
    def compute_stats(times_us: list[float]) -> dict:
        """Compute summary statistics from a list of per-iteration microsecond
        latencies.  Min is the primary metric (CUTLASS convention)."""
        if not times_us:
            return {"min": 0.0, "median": 0.0, "mean": 0.0,
                    "stddev": 0.0, "p99": 0.0}
        sorted_times = sorted(times_us)
        n = len(sorted_times)
        mean = sum(sorted_times) / n
        variance = sum((x - mean) ** 2 for x in sorted_times) / n
        p99_idx = max(0, int(math.ceil(n * 0.99)) - 1)
        return {
            "min": sorted_times[0],
            "median": statistics.median(sorted_times),
            "mean": mean,
            "stddev": math.sqrt(variance),
            "p99": sorted_times[p99_idx],
        }


# ---------------------------------------------------------------------------
# core benchmark loop
# ---------------------------------------------------------------------------

def run_benchmark(
    config: BenchmarkConfig,
    setup_fn: Callable[[], Any],
    llsys_fn: Callable[[Any], None],
    ref_fn: Optional[Callable[[Any], None]] = None,
) -> BenchmarkResult:
    """Run a timed benchmark with warmup + repeat cycles using GPU-side timers.

    Parameters
    ----------
    config : BenchmarkConfig
        Operator name, shape, dtype, warmup/repeat counts, and byte/flop counts.
    setup_fn : () -> Any
        Called once per iteration to produce fresh buffers (fed to both
        *llsys_fn* and *ref_fn*).
    llsys_fn : (buffers) -> None
        The LLAISYS operator under test.  Receives the return value of
        *setup_fn*.
    ref_fn : (buffers) -> None or None
        Optional reference implementation (e.g. PyTorch).  Measured with the
        same warmup + repeat protocol.

    Returns
    -------
    BenchmarkResult
    """
    # ---- helper: time a single function -----------------------------------
    def _measure(fn: Callable[[Any], None]) -> list[float]:
        # warmup
        for _ in range(config.warmup):
            fn(setup_fn())
        torch.cuda.synchronize()

        # timed iterations
        times_us: list[float] = []
        timer = CUDATimer()
        for _ in range(config.repeat):
            buffers = setup_fn()
            timer.record_start()
            fn(buffers)
            timer.record_end()
            times_us.append(timer.elapsed_ms() * 1000.0)
        return times_us

    # ---- LLAISYS measurement ----------------------------------------------
    llsys_times = _measure(llsys_fn)
    stats = BenchmarkResult.compute_stats(llsys_times)
    elapsed_s = stats["min"] / 1e6

    total_bytes = config.total_bytes_read + config.total_bytes_write
    bandwidth = (total_bytes / elapsed_s / 1e9) if elapsed_s > 0 else 0.0
    tflops = (config.total_flops / elapsed_s / 1e12) if elapsed_s > 0 else 0.0

    result = BenchmarkResult(
        operator=config.op_name,
        dtype=config.dtype_name,
        shape=config.shape,
        latency_us=stats["min"],
        latency_stats=stats,
        bandwidth_GBs=bandwidth,
        TFLOPS=tflops,
    )

    # ---- reference baseline -----------------------------------------------
    if ref_fn is not None:
        ref_times = _measure(ref_fn)
        ref_stats = BenchmarkResult.compute_stats(ref_times)
        ref_elapsed_s = ref_stats["min"] / 1e6
        ref_tflops = (config.total_flops / ref_elapsed_s / 1e12) if ref_elapsed_s > 0 else 0.0

        result.baseline_name = "pytorch"
        result.baseline_latency_us = ref_stats["min"]
        result.baseline_TFLOPS = ref_tflops
        result.speedup = (ref_stats["min"] / stats["min"]) if stats["min"] > 0 else 0.0

    return result


# ---------------------------------------------------------------------------
# terminal output
# ---------------------------------------------------------------------------

def _shape_str(r: BenchmarkResult) -> str:
    return " x ".join(str(v) for v in r.shape.values())

def format_summary(result: BenchmarkResult) -> str:
    """Silent during benchmark loop — table printed by format_results_table()."""
    return ""

def format_results_table(results: list[BenchmarkResult],
                          shape_label: str = "shape") -> str:
    """Aligned table output for a batch of results."""
    if not results:
        return "(no results)"

    has_baseline = any(r.baseline_name for r in results)
    bl_name = next((r.baseline_name for r in results if r.baseline_name), "")

    # dynamic widths
    shape_w = max(len(_shape_str(r)) for r in results)
    dtype_w = max(len(r.dtype) for r in results)
    _S = "  "

    # column headers
    ll_hdr = "latency(us)"
    tf_hdr = "TFLOPS"
    LLA_COLS = [ll_hdr, tf_hdr]
    bl_hdr_us = "baseline(us)"
    bl_hdr_tf = "baseline(TF)"
    sp_hdr = "speedup"
    BL_COLS = [bl_hdr_us, bl_hdr_tf, sp_hdr]

    val_w = max(len(ll_hdr), len(tf_hdr), 10)
    bl_w  = max(len(bl_hdr_us), len(bl_hdr_tf), 10) if has_baseline else 0
    sp_w  = len(sp_hdr) if has_baseline else 0

    # sample row for total width (includes #0  prefix)
    sample = ("  0 " + _S + "x" * shape_w + _S + "x" * dtype_w +
              _S + "x" * val_w + _S + "x" * val_w)
    if has_baseline:
        sample += _S + "x" * bl_w + _S + "x" * bl_w + _S + "x" * sp_w
    W = len(sample)

    lines = [f"\n  Operator: {results[0].operator}", f"  {'═' * W}"]

    # header
    hdr = (f"  {'  ':<3s}{shape_label:<{shape_w}}{_S}{'dtype':<{dtype_w}}{_S}"
           f"{ll_hdr:>{val_w}}{_S}{tf_hdr:>{val_w}}")
    if has_baseline:
        hdr += f"{_S}{bl_hdr_us:>{bl_w}}{_S}{bl_hdr_tf:>{bl_w}}{_S}{sp_hdr:>{sp_w}}"
    lines.append(hdr)
    lines.append(f"  {'─' * W}")

    # data rows
    latencies = []
    for i, r in enumerate(results):
        s = _shape_str(r)
        row = (f"  {i:<3d} {s:<{shape_w}}{_S}{r.dtype:<{dtype_w}}{_S}"
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

    # footer
    if latencies:
        lines.append(f"  {'─' * W}")
        foot = (f"  {results[0].dtype}"
                f"  │ range: {min(latencies):.1f} — {max(latencies):.1f} us"
                f"  │ mean: {sum(latencies)/len(latencies):.1f} us")
        if has_baseline:
            sp = [r.speedup for r in results if r.baseline_latency_us > 0]
            if sp:
                foot += f"  │ speedup: {min(sp):.2f}x — {max(sp):.2f}x"
        lines.append(foot)

    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# JSON persistence
# ---------------------------------------------------------------------------

def save_results(results: list[BenchmarkResult], output_dir: str,
                  shape_label: str = "shape") -> str:
    """Write results to a timestamped JSON file, symlink latest.json, and
    print a terminal summary."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # environment info
    gpu_model = ""
    cuda_version = ""
    try:
        gpu_model = torch.cuda.get_device_name(0)
    except Exception:
        gpu_model = "unknown"
    try:
        cuda_version = torch.version.cuda or ""
    except Exception:
        cuda_version = "unknown"

    payload = {
        "environment": {
            "cuda_version": cuda_version,
            "gpu_model": gpu_model,
            "timestamp": timestamp,
        },
        "results": [r.to_dict() for r in results],
    }

    json_path = os.path.join(run_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # symlink latest.json -> run_dir/results.json
    latest_link = os.path.join(output_dir, "latest.json")
    if os.path.islink(latest_link) or os.path.exists(latest_link):
        os.remove(latest_link)
    try:
        os.symlink(json_path, latest_link)
    except OSError:
        # symlink not supported on this platform (e.g. Windows without
        # developer mode); fall back to copying the file.
        import shutil
        shutil.copy2(json_path, latest_link)

    # terminal summary
    print(f"\n  GPU: {gpu_model}  |  CUDA: {cuda_version}  |  Saved: {os.path.relpath(json_path)}")
    print(format_results_table(results, shape_label))

    return json_path
