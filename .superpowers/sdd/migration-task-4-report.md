# Migration Task 4 Report — Python WeightRole mapping (Scheme A)

**Date:** 2026-08-16  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Base:** `f4edc23`  
**Commit:** `b868f0d` — `feat: sync qwen2 weights into model context via roles`

## Summary

Scheme A compat shell: Python still fills `LlaisysQwen2Weights` via HF-name mapping; C++ `sync_weights_from_legacy_struct()` copies into `ModelContext` through `Model::set_weight` / `WeightRole`. Public `generate` / server API unchanged. Default forward remains **legacy** (Task 5 not flipped).

## Changes

| Path | Role |
|---|---|
| `python/llaisys/models/qwen2_weight_map.py` | HF name → `(WeightRole, layer)` + legacy-slot assign helper |
| `python/llaisys/models/qwen2.py` | `_assign_weight` driven by the map; tie `out_embed=in_embed` unchanged |
| `src/models/qwen2.hpp/.cpp` | `sync_weights_from_legacy_struct()`; defer `start()` until first `submit()` so sync runs after Python load; `forward_layers` re-syncs for C++ dual-path tests |
| `test/test_qwen2_weight_map.py` | DeepSeek critical keys (bias + tie doc) |

## Verification

```bash
PYTHONPATH=./python:$PYTHONPATH python test/test_qwen2_weight_map.py
→ OK (4 tests)

XMAKE_ROOT=y xmake && xmake install
XMAKE_ROOT=y xmake run llaisys-qwen2-dual-path-nvidia-test
→ qwen2 dual path NVIDIA test passed

PYTHONPATH=./python:$PYTHONPATH python test/test_qwen2_layer_parity.py \
  --device nvidia \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --max_steps 32 --skip-hf
→ HARD GATE PASSED (equal_token_count=42)
```

Token sequences match Task 3 evidence (same greedy 42-token list).

## Concerns / follow-ups

- Worker starts lazily on first `submit` (constructor no longer starts). Dual-path C++ tests already `stop()` + call `forward_*` directly; they rely on sync inside `forward_layers`.
- Sync still runs every `forward_layers` call (cheap shared_ptr fills) so late weight mutation in unit tests stays correct; production path also syncs once in `start()`.
- Scheme B (`llaisysQwen2SetWeight`) not added; optional later.
- Do **not** flip default to layer (Task 5).
