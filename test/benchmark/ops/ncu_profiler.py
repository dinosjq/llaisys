"""
NCU (NVIDIA Nsight Compute) integration module for CUDA operator profiling.

Provides NCU binary discovery, subprocess double-launch pattern (launch the
application under ``ncu`` to capture a ``.ncu-rep`` report, then extract
metrics from that report as CSV), bottleneck classification from weighted
metric aggregation, and a high-level ``profile_with_ncu()`` entry point.

Usage::

    from ncu_profiler import NCUConfig, profile_with_ncu

    config = NCUConfig(set="detailed", launch_skip=5, launch_count=3)
    report = profile_with_ncu(
        benchmark_script="my_op_bench.py",
        op_args=["--shape", "4096x4096", "--dtype", "f16"],
        ncu_config=config,
        output_dir="./profiles",
    )
    print(report.bottleneck, report.sm_throughput_pct)
"""

import csv
import io
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------

@dataclass
class NCUConfig:
    """Configuration for a single NCU profiling session.

    Attributes
    ----------
    set : str
        NCU section set: ``"quick"`` (default), ``"detailed"``, or ``"full"``.
    clock_control : str
        Clock control mode (``"base"``, ``"none"``, ``"noboost"``).
    launch_skip : int
        Number of kernel launches to skip before profiling.
    launch_count : int
        Number of kernel launches to profile.
    kernel_name : Optional[str]
        If set, filter profiling to this kernel regex.
    replay_mode : str
        NCU replay mode (``"kernel"``, ``"application"``, ``"range"``).
    metrics : list
        NCU metric names to collect.
    """

    set: str = "detailed"
    clock_control: str = "base"
    launch_skip: int = 10
    launch_count: int = 1
    kernel_name: Optional[str] = None
    replay_mode: str = "kernel"
    metrics: list = field(default_factory=lambda: [
        "gpu__time_duration.sum",
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
    ])

    def to_cli_args(self) -> list[str]:
        """Return the CLI argument list equivalent of this config."""
        args = [
            "--set", self.set,
            "--replay-mode", self.replay_mode,
            "--launch-skip", str(self.launch_skip),
            "--launch-count", str(self.launch_count),
            "--clock-control", self.clock_control,
            "--target-processes", "all",
        ]
        if self.kernel_name is not None:
            args.extend(["--kernel-name", self.kernel_name])
        return args


# ---------------------------------------------------------------------------
# result
# ---------------------------------------------------------------------------

@dataclass
class ProfileReport:
    """Aggregated NCU profiling report for a single operator.

    Attributes
    ----------
    operator : str
        Operator name (e.g. ``"linear"``).
    ncu_set : str
        Which NCU section set was used.
    report_file : str
        Path to the ``.ncu-rep`` report file.
    kernel_count : int
        Number of distinct kernels captured.
    total_time_us : float
        Total GPU time across all kernels (microseconds).
    sm_throughput_pct : float
        Weighted SM throughput (% of peak sustained elapsed).
    dram_throughput_pct : float
        Weighted DRAM throughput (% of peak sustained elapsed).
    occupancy_pct : float
        Weighted warp occupancy (% of peak sustained elapsed).
    bottleneck : str
        Classified bottleneck category.
    raw_metrics : list[dict]
        Raw per-kernel metrics as returned by ``extract_metrics_from_report()``.
    """

    operator: str
    ncu_set: str
    report_file: str
    kernel_count: int
    total_time_us: float
    sm_throughput_pct: float
    dram_throughput_pct: float
    occupancy_pct: float
    bottleneck: str
    raw_metrics: list[dict]


# ---------------------------------------------------------------------------
# NCU binary discovery
# ---------------------------------------------------------------------------

_NCU_DOWNLOAD_URL = "https://developer.nvidia.com/nsight-compute"

# Common installation paths searched when ncu is not on PATH.
# The last two entries cover WSL-accessing-Windows-side NCU installations.
_KNOWN_NCU_PATHS: list[str] = [
    "/usr/local/cuda/bin/ncu",
    "/opt/cuda/bin/ncu",
    "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.bat",
    "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.exe",
]


def find_ncu() -> str:
    """Locate the ``ncu`` binary on the system.

    Searches ``PATH`` first (via ``shutil.which``), then falls back to
    common installation directories.

    Returns
    -------
    str
        Absolute or relative path to the ``ncu`` executable.

    Raises
    ------
    FileNotFoundError
        If ``ncu`` cannot be found in any known location.
    """
    resolved = shutil.which("ncu")
    if resolved is not None:
        return resolved

    for cand in _KNOWN_NCU_PATHS:
        if os.path.isfile(cand):
            return cand

    raise FileNotFoundError(
        "NCU binary not found.  Install NVIDIA Nsight Compute from:\n"
        f"  {_NCU_DOWNLOAD_URL}\n"
        "Make sure 'ncu' is on PATH or in one of the known locations:\n  "
        + "\n  ".join(_KNOWN_NCU_PATHS)
    )


# ---------------------------------------------------------------------------
# subprocess helpers
# ---------------------------------------------------------------------------

def run_under_ncu(
    ncu_binary: str,
    app_cmd: list[str],
    config: NCUConfig,
    output_file: str,
    *,
    timeout: int = 600,
) -> subprocess.CompletedProcess:
    """Launch *app_cmd* under ``ncu`` and wait for completion.

    The output ``.ncu-rep`` file is written to *output_file* (without the
    ``.ncu-rep`` extension -- NCU appends it automatically).

    Parameters
    ----------
    ncu_binary : str
        Path to the ``ncu`` executable (as returned by :func:`find_ncu`).
    app_cmd : list[str]
        The application command-line (argv) to profile.
    config : NCUConfig
        NCU configuration controlling section set, replay mode, etc.
    output_file : str
        Base path for the output report.  NCU will actually write
        ``<output_file>.ncu-rep``.
    timeout : int
        Subprocess timeout in seconds (default 600).

    Returns
    -------
    subprocess.CompletedProcess
        The completed subprocess result (stdout/stderr captured as text).
    """
    cmd: list[str] = [ncu_binary] + config.to_cli_args() + [
        "-o", output_file,
        "-f",
    ] + app_cmd

    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


# ---------------------------------------------------------------------------
# CSV parsing
# ---------------------------------------------------------------------------

def parse_ncu_csv(csv_text: str) -> list[dict]:
    """Parse NCU ``--csv --page raw`` output into a list of per-kernel dicts.

    The NCU raw CSV format has three logical sections:

    * Row 0 -- metric names (headers).
    * Row 1 -- units (ignored).
    * Row 2+ -- one row per kernel invocation, values aligned with headers.

    Numeric values are converted to ``float`` where possible; non-numeric
    strings (e.g. kernel name) are left as-is.

    Parameters
    ----------
    csv_text : str
        Raw CSV text from NCU stdout or a ``.csv`` file.

    Returns
    -------
    list[dict]
        One dict per kernel row, mapping ``header_name -> value``.
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        return []

    reader = csv.reader(io.StringIO("\n".join(lines)))
    headers = next(reader, None)
    if headers is None:
        return []

    # Row 1 = units -- read and discard.
    _units = next(reader, None)

    rows: list[dict] = []
    for row in reader:
        if not row:
            continue
        entry: dict[str, Any] = {}
        for i, val in enumerate(row):
            if i >= len(headers):
                break
            key = headers[i]
            # Attempt numeric conversion.
            try:
                entry[key] = float(val)
            except (ValueError, TypeError):
                entry[key] = val
        if entry:
            rows.append(entry)

    return rows


# ---------------------------------------------------------------------------
# report extraction
# ---------------------------------------------------------------------------

def extract_metrics_from_report(
    ncu_binary: str,
    ncu_rep_file: str,
    metrics: list[str],
) -> list[dict]:
    """Extract selected metrics from an existing ``.ncu-rep`` report file.

    Invokes ``ncu -i <rep> --csv --page raw --metrics <m1,m2,...>``
    and parses the CSV output.

    Parameters
    ----------
    ncu_binary : str
        Path to the ``ncu`` executable.
    ncu_rep_file : str
        Path to the ``.ncu-rep`` report file to read.
    metrics : list[str]
        Metric names to extract.

    Returns
    -------
    list[dict]
        Parsed per-kernel metric rows (see :func:`parse_ncu_csv`).
    """
    cmd: list[str] = [
        ncu_binary,
        "-i", ncu_rep_file,
        "--csv",
        "--page", "raw",
        "--metrics", ",".join(metrics),
    ]
    raw = subprocess.check_output(cmd, text=True, timeout=300)
    return parse_ncu_csv(raw)


# ---------------------------------------------------------------------------
# bottleneck classification
# ---------------------------------------------------------------------------

def classify_bottleneck(kernel_data: list[dict]) -> dict:
    """Classify the GPU bottleneck from per-kernel NCU metrics.

    Uses a weighted average over kernels, where each kernel's contribution
    is weighted by ``gpu__time_duration.sum``.

    Classification rules (thresholds applied to the weighted percentages):

    +----------------------------+---------------------------------------+
    | Condition                  | Bottleneck                            |
    +============================+=======================================+
    | DRAM > 60 AND SM < 20      | ``"memory_bound"``                    |
    +----------------------------+---------------------------------------+
    | SM > 60 AND DRAM < 20      | ``"compute_bound"``                   |
    +----------------------------+---------------------------------------+
    | both < 30                  | ``"launch_occupancy_bound"``          |
    +----------------------------+---------------------------------------+
    | otherwise                  | ``"latency_bound"``                   |
    +----------------------------+---------------------------------------+

    Parameters
    ----------
    kernel_data : list[dict]
        Per-kernel metric rows (from :func:`extract_metrics_from_report`
        or :func:`parse_ncu_csv`).

    Returns
    -------
    dict
        Keys: ``bottleneck``, ``sm_throughput_pct``, ``dram_throughput_pct``,
        ``occupancy_pct``.
    """
    SM_KEY = "sm__throughput.avg.pct_of_peak_sustained_elapsed"
    DRAM_KEY = "dram__throughput.avg.pct_of_peak_sustained_elapsed"
    OCC_KEY = "sm__warps_active.avg.pct_of_peak_sustained_elapsed"
    TIME_KEY = "gpu__time_duration.sum"

    if not kernel_data:
        return {
            "bottleneck": "unknown",
            "sm_throughput_pct": 0.0,
            "dram_throughput_pct": 0.0,
            "occupancy_pct": 0.0,
        }

    total_time = 0.0
    weighted_sm = 0.0
    weighted_dram = 0.0
    weighted_occ = 0.0

    for k in kernel_data:
        duration = float(k.get(TIME_KEY, 0.0))
        sm = float(k.get(SM_KEY, 0.0))
        dram = float(k.get(DRAM_KEY, 0.0))
        occ = float(k.get(OCC_KEY, 0.0))

        weighted_sm += sm * duration
        weighted_dram += dram * duration
        weighted_occ += occ * duration
        total_time += duration

    if total_time > 0.0:
        sm_pct = weighted_sm / total_time
        dram_pct = weighted_dram / total_time
        occ_pct = weighted_occ / total_time
    else:
        sm_pct = 0.0
        dram_pct = 0.0
        occ_pct = 0.0

    # Classification rules.
    if dram_pct > 60.0 and sm_pct < 20.0:
        bottleneck = "memory_bound"
    elif sm_pct > 60.0 and dram_pct < 20.0:
        bottleneck = "compute_bound"
    elif sm_pct < 30.0 and dram_pct < 30.0:
        bottleneck = "launch_occupancy_bound"
    else:
        bottleneck = "latency_bound"

    return {
        "bottleneck": bottleneck,
        "sm_throughput_pct": sm_pct,
        "dram_throughput_pct": dram_pct,
        "occupancy_pct": occ_pct,
    }


# ---------------------------------------------------------------------------
# high-level entry point
# ---------------------------------------------------------------------------

def profile_with_ncu(
    benchmark_script: str,
    op_args: list[str],
    ncu_config: NCUConfig,
    output_dir: str,
    *,
    operator_name: str = "",
) -> ProfileReport:
    """Profile a benchmark script under NCU and return a structured report.

    This is the recommended entry point for on-demand profiling from
    benchmark scripts.  It follows the **double-launch** pattern:

    1. Run the benchmark script under ``ncu`` to produce a ``.ncu-rep`` file.
    2. Extract selected metrics from the ``.ncu-rep`` as CSV.
    3. Classify the bottleneck from the extracted metrics.

    The benchmark script is invoked with an appended ``--ncu-child`` flag so
    it can skip internal NCU instrumentation (the "child" half of the
    double-launch).

    Parameters
    ----------
    benchmark_script : str
        Path to the benchmark Python script (e.g. ``"ops/linear.py"``).
    op_args : list[str]
        CLI arguments forwarded to the benchmark script.
    ncu_config : NCUConfig
        NCU section set, launch control, metric list, etc.
    output_dir : str
        Directory where ``.ncu-rep`` reports are written.
    operator_name : str
        Human-readable operator name for the report (default: basename of
        *benchmark_script* without extension).

    Returns
    -------
    ProfileReport

    Raises
    ------
    FileNotFoundError
        If the ``ncu`` binary cannot be found (see :func:`find_ncu`).
    subprocess.CalledProcessError
        If the profiled application exits non-zero.
    RuntimeError
        If the ``.ncu-rep`` report file is missing after the NCU run.
    """
    os.makedirs(output_dir, exist_ok=True)

    # -- determine operator name ----------------------------------------------
    if not operator_name:
        operator_name = os.path.splitext(os.path.basename(benchmark_script))[0]

    # -- locate NCU -----------------------------------------------------------
    ncu_binary = find_ncu()

    # -- launch application under NCU -----------------------------------------
    output_base = os.path.join(output_dir, f"profile_{operator_name}")
    app_cmd: list[str] = [sys.executable, benchmark_script, "--ncu-child"] + op_args

    proc = run_under_ncu(
        ncu_binary=ncu_binary,
        app_cmd=app_cmd,
        config=ncu_config,
        output_file=output_base,
    )

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(
            proc.returncode,
            [ncu_binary] + ncu_config.to_cli_args() + ["-o", output_base, "-f"] + app_cmd,
            output=proc.stdout,
            stderr=proc.stderr,
        )

    rep_file = output_base + ".ncu-rep"
    if not os.path.isfile(rep_file):
        raise RuntimeError(
            f"NCU report file not found after profiling: {rep_file}\n"
            f"ncu stdout:\n{proc.stdout}\n"
            f"ncu stderr:\n{proc.stderr}"
        )

    # -- extract metrics ------------------------------------------------------
    raw_metrics = extract_metrics_from_report(
        ncu_binary=ncu_binary,
        ncu_rep_file=rep_file,
        metrics=ncu_config.metrics,
    )

    # -- classify bottleneck --------------------------------------------------
    bottleneck_info = classify_bottleneck(raw_metrics)

    # -- compute total time ---------------------------------------------------
    total_time_us = 0.0
    for k in raw_metrics:
        total_time_us += float(k.get("gpu__time_duration.sum", 0.0))

    return ProfileReport(
        operator=operator_name,
        ncu_set=ncu_config.set,
        report_file=rep_file,
        kernel_count=len(raw_metrics),
        total_time_us=total_time_us,
        sm_throughput_pct=bottleneck_info["sm_throughput_pct"],
        dram_throughput_pct=bottleneck_info["dram_throughput_pct"],
        occupancy_pct=bottleneck_info["occupancy_pct"],
        bottleneck=bottleneck_info["bottleneck"],
        raw_metrics=raw_metrics,
    )


# ---------------------------------------------------------------------------
# high-level --use-ncu entry point
# ---------------------------------------------------------------------------

_NCU_METRICS = [
    "gpu__time_duration.sum",
    "dram__bytes_read.sum",
    "dram__bytes_write.sum",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
]


def profile_benchmark(script: str, argv: list[str], op_name: str,
                      example_indices: Optional[list[int]] = None) -> None:
    """Profile one or more shapes under NCU.

    1. Strips ``--use-ncu`` / ``--example-index`` from *argv*.
    2. Runs the benchmark normally for each requested shape (limited via
       ``--M`` / ``--variant`` or ``--example-index``).
    3. Parses each ``.ncu-rep``, classifies the bottleneck, and writes
       ``ncu_results.json`` into the script's results directory.

    *example_indices* of ``None`` means auto-select: the benchmark is
    run once to collect speedup data, then the worst-performing shape
    (lowest speedup) is profiled.
    """
    import json as _json, time as _time

    # --- strip NCU-only flags from argv ------------------------------------
    stripped = _strip_ncu_flags(argv)

    # --- locate NCU --------------------------------------------------------
    ncu_binary = find_ncu()

    script_dir = os.path.dirname(os.path.abspath(script))
    out_dir = os.path.join(script_dir, "results", "ncu")
    os.makedirs(out_dir, exist_ok=True)

    results_dir = os.path.join(script_dir, "results")
    ncu_results: list[dict] = []

    # --- determine indices -------------------------------------------------
    if example_indices is None:
        # auto-select: run once, pick worst speedup
        idx = _auto_select_index(stripped, script, results_dir)
        example_indices = [idx]

    for idx in example_indices:
        shape_args = _shape_args_for_index(idx, stripped, script)
        child_cmd = [sys.executable, script] + shape_args

        rep_base = os.path.join(out_dir, f"{op_name}_idx{idx}")
        rep_file = rep_base + ".ncu-rep"

        ncu_cmd = [ncu_binary,
                   "--set", "full",
                   "--clock-control", "base",
                   "--target-processes", "all",
                   "-o", rep_base,
                   "-f",
                   "--"] + child_cmd

        proc = subprocess.run(ncu_cmd, capture_output=True, text=True,
                              timeout=600)

        if proc.returncode != 0:
            print(f"  NCU failed (exit {proc.returncode}):\n{proc.stderr}")
            sys.exit(1)

        if not os.path.isfile(rep_file):
            print(f"  NCU report not generated: {rep_file}\n{proc.stderr}")
            sys.exit(1)

        # --- parse & classify ----------------------------------------------
        raw = extract_metrics_from_report(ncu_binary, rep_file, _NCU_METRICS)
        classif = classify_bottleneck(raw)

        total_ns = sum(float(k.get("gpu__time_duration.sum", 0)) for k in raw)
        entry = {
            "operator": op_name,
            "shape_index": idx,
            "report_file": rep_file,
            "kernel_count": len(raw),
            "total_time_us": total_ns / 1000.0,
            "bottleneck": classif["bottleneck"],
            "sm_throughput_pct": classif["sm_throughput_pct"],
            "dram_throughput_pct": classif["dram_throughput_pct"],
            "occupancy_pct": classif["occupancy_pct"],
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        ncu_results.append(entry)
        print(f"  NCU #{idx}: {classif['bottleneck']}  "
              f"SM={classif['sm_throughput_pct']:.1f}%  "
              f"DRAM={classif['dram_throughput_pct']:.1f}%  "
              f"Occ={classif['occupancy_pct']:.1f}%")

    # --- persist -----------------------------------------------------------
    json_path = os.path.join(results_dir, "ncu_results.json")
    with open(json_path, "w") as f:
        _json.dump(ncu_results, f, indent=2)
    print(f"  NCU results saved: {json_path}")


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _strip_ncu_flags(argv: list[str]) -> list[str]:
    """Remove --use-ncu and --example-index and its value from argv."""
    out = []
    skip = False
    for a in argv:
        if skip:
            skip = False
            continue
        if a == "--use-ncu":
            continue
        if a in ("--example-index", "--example-index"):
            skip = True
            continue
        if a.startswith("--example-index="):
            continue
        out.append(a)
    return out


def _auto_select_index(stripped_argv: list[str], script: str,
                       results_dir: str) -> int:
    """Run benchmark once, parse speedup data, return worst-performing index."""
    import json as _json, tempfile

    tmpdir = tempfile.mkdtemp(prefix="_ncu_auto_")
    auto_cmd = [sys.executable, script] + stripped_argv + \
               ["--output", tmpdir]
    try:
        subprocess.run(auto_cmd, capture_output=True, timeout=300)
        latest = os.path.join(tmpdir, "latest.json")
        if not os.path.isfile(latest):
            return 0
        with open(latest) as f:
            data = _json.load(f)
        results = data.get("results", [])
        if not results:
            return 0
        speeds = [(i, r.get("speedup", 1.0)) for i, r in enumerate(results)]
        below = [(i, s) for i, s in speeds if s < 1.0]
        if below:
            return min(below, key=lambda x: x[1])[0]
        return min(speeds, key=lambda x: abs(x[1] - 1.0))[0]
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def _shape_args_for_index(index: int, stripped_argv: list[str],
                          script: str) -> list[str]:
    """Build CLI args to limit the benchmark to a single shape by index."""
    script_name = os.path.basename(script)

    if script_name == "benchmark_linear.py":
        M_VALUES = [1, 16, 64, 128, 512, 2048]
        VARIANTS = ["q_proj", "gate_proj", "down_proj"]
        total = len(VARIANTS) * len(M_VALUES)
        if index < 0 or index >= total:
            raise ValueError(f"Index {index} out of range (0-{total - 1})")
        vi = index // len(M_VALUES)
        mi = index % len(M_VALUES)
        return ["--variant", VARIANTS[vi], "--M", str(M_VALUES[mi])]

    # non-linear scripts: pass --example-index to select shape
    return ["--example-index", str(index)]

