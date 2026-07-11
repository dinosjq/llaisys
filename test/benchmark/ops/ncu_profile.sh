#!/bin/bash
# NCU profiling convenience wrapper for operator benchmarks.
# Usage: bash ncu_profile.sh --op linear --set detailed --dtype f16
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="${SCRIPT_DIR}/results/ncu"
mkdir -p "${OUT_DIR}"

NCU=$(which ncu 2>/dev/null || echo "")
if [ -z "$NCU" ]; then
    for candidate in \
        "/usr/local/cuda/bin/ncu" \
        "/opt/cuda/bin/ncu" \
        "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.bat" \
        "/mnt/c/Program Files/NVIDIA Corporation/Nsight Compute 2022.2.0/ncu.exe"; do
        if [ -f "$candidate" ]; then NCU="$candidate"; break; fi
    done
fi

if [ -z "$NCU" ]; then
    echo "ERROR: ncu not found. Install Nsight Compute or add it to PATH."
    exit 1
fi

OP="linear"
NCU_SET="detailed"
DTYPE="f16"

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
echo "NCU: ${NCU}"
echo "Script: ${BENCH_SCRIPT}"
echo "Output: ${OUT_DIR}/${OP}_${NCU_SET}.ncu-rep"

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
