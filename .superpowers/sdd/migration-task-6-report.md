# Migration Task 6 Report — P0/D0/D1 performance gate

**Date:** 2026-08-16  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Model:** `/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B`  
**GPU:** NVIDIA GeForce RTX 4060 Laptop GPU  

## gate_status: passed

Median `wall_per_output_token` (ms/tok) after shape-matched warmup; threshold ≤ **3%** regression (layer vs legacy).

| Shape | Spec | median legacy | median layer | relative | Result |
|---|---|---:|---:|---:|---|
| P0 | input 256 / gen 1 | 47.364 | 47.234 | **-0.27%** | PASS |
| D0 | input 1 / gen 128 | 16.381 | 16.041 | **-2.08%** | PASS |
| D1 | input 1024 / gen 128 | 18.302 | 18.487 | **+1.01%** | PASS |

**overall_pass:** true — keep default Layer path (`use_layer_forward_=true`).

## Protocol

- Script: `test/profile/migration_perf_gate.py`
- Env: `LLAISYS_QWEN2_LAYER_FORWARD=0` (legacy) vs `=1` (layer); subprocess-isolated runs
- Repeats: 3 per shape per mode; median of `wall_per_output_token_ms`
- Artifact: `test/results/migration_perf_gate_20260816_020601.json`

```bash
PYTHONPATH=./python:$PYTHONPATH python test/profile/migration_perf_gate.py \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --shapes P0,D0,D1 --repeats 3 \
  --out test/results/migration_perf_gate_YYYYMMDD_HHMMSS.json
```

## Notes

- Earlier Task 3 note of “~3× slower” was not reproduced under this steady wall-clock protocol (likely capture / cold-path noise).
- Full nsys timeline optional for kernel attribution; gate metric is wall ms/tok as specified in the migration plan.
- Rollback remains: `LLAISYS_QWEN2_LAYER_FORWARD=0`.
