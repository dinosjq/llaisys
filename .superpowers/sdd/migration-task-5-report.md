# Migration Task 5 Report — Default Qwen2 to layer stack

**Date:** 2026-08-16  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Base:** `933e3b5`  
**Commit:** `cb9893e` — `feat: default qwen2 inference to layer stack`

## Summary

Production Qwen2 inference now defaults to the layer-stack forward path. Legacy hard-coded `forward_legacy` remains available for rollback via environment variable.

## Rollback

```bash
LLAISYS_QWEN2_LAYER_FORWARD=0   # force legacy
# unset / empty / any other value → layer (default)
```

Also noted in `python/llaisys/server.py` module docstring and `docs/superpowers/plans/2026-08-15-qwen2-framework-migration.md`.

## Changes

| Path | Role |
|---|---|
| `src/models/qwen2.cpp` | `env_use_layer_forward()`: unset → `true`; `=0` → legacy |
| `src/models/qwen2.hpp` | `use_layer_forward_ = true`; comments |
| `test/models/qwen2_dual_path_nvidia_test.cpp` | default=layer + env=0 forces legacy |
| `python/llaisys/server.py` | rollback note in usage docstring |
| migration plan | Task 5 steps checked; rollback line |

## Verification

```bash
XMAKE_ROOT=y xmake && xmake install
XMAKE_ROOT=y xmake run llaisys-qwen2-dual-path-nvidia-test
→ qwen2 dual path NVIDIA test passed

XMAKE_ROOT=y xmake run llaisys-prepare-parity-test
→ prepare parity test passed

XMAKE_ROOT=y xmake run llaisys-model-layer-cpu-test
→ qwen2 embedding and ffn CPU layer tests passed

XMAKE_ROOT=y xmake run llaisys-model-layer-nvidia-test
→ qwen2 NVIDIA layer stack capture test passed

PYTHONPATH=./python:$PYTHONPATH python3 test/test_qwen2_layer_parity.py \
  --device nvidia --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --max_steps 32 --skip-hf
→ HARD GATE PASSED (equal_token_count=41)

PYTHONPATH=./python:$PYTHONPATH python3 test/test_infer.py \
  --device nvidia --model ... --test --max_step 8
→ OK (default path = layer)

PYTHONPATH=./python:$PYTHONPATH python3 test/test_server.py \
  --model ... --device nvidia --port 18765
→ All server tests passed
```

## Concerns / follow-ups

- Task 6: P0/D0/D1 performance gate (≤3% vs explicit legacy) still outstanding; if regression >3%, flip default back to legacy.
- Parity token count this run was 41 (Task 3/4 reported 42); sequences still match legacy↔layer under the same flags — not a gate failure.
