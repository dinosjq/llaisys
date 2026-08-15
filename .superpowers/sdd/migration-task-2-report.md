# Migration Task 2 Report — Qwen2 dual forward path

**Date:** 2026-08-15  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Base:** `70898c4`  
**Commit:** `c3fb5e6` — `feat: add qwen2 dual forward path behind flag`

## Summary

Qwen2 now holds a `framework::ModelContext`, binds legacy weights/KV/workspace into it, and exposes dual forward paths. Production default remains hardcoded `forward_legacy`. Layer path is gated by `use_layer_forward_` / env `LLAISYS_QWEN2_LAYER_FORWARD` (unset or `0` → false).

## Behavior

| Entry | Role |
|---|---|
| `forward` | Dispatches by `use_layer_forward_` |
| `forward_legacy` | Previous hardcoded loop (unchanged logic; sampling extracted) |
| `forward_layers` | `bind_context_*` → `run_qwen2_layer_stack` → same `sample_from_logits` |
| Worker | `use_layer_forward_ ? forward_layers : forward_legacy` |

Stack already performs CPU `stable_capture` after layer-0 attn/FFN residuals. Dual-path NVIDIA test covers prefill and decode-after-prefill logits + greedy tokens.

## Files

- `src/models/qwen2.hpp` / `qwen2.cpp` — context, binds, dual path, env flag
- `test/models/qwen2_dual_path_nvidia_test.cpp` — tiny NVIDIA fixture parity
- `xmake.lua` — target `llaisys-qwen2-dual-path-nvidia-test`

## Verification

```text
XMAKE_ROOT=y xmake run llaisys-qwen2-dual-path-nvidia-test
→ qwen2 dual path NVIDIA test passed

XMAKE_ROOT=y xmake run llaisys-model-layer-nvidia-test
→ qwen2 NVIDIA layer stack capture test passed

XMAKE_ROOT=y xmake run llaisys-prepare-parity-test
→ prepare parity test passed
```

## Concerns / follow-ups

- Weights still live in legacy `Qwen2Weights`; bind copies `tensor_t` into context each `forward_layers` (Task 4 sync/role upload).
- Dual-path test links static archives with explicit order (`inherit=false`) due to core↔ops cycle.
- Real-model greedy gate is Task 3; default must stay legacy until then.
