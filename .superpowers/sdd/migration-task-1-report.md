# Migration Task 1 Report — lift Qwen2 `prepare_*` into Model

**Date:** 2026-08-15  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Base:** `100d324`  
**Commit:** `70898c4` — `refactor: lift qwen2 prepare helpers into Model`

## Summary

Moved `prepare_block_table` / `prepare_prefill` / `prepare_decode` from `Qwen2` into `llaisys::framework::Model` as static helpers. `Qwen2` methods are thin forwards. Return type remains `Qwen2Pack` (not `BatchRuntime`) per brief. Production `forward` path unchanged (still hardcoded; no dual-path).

## TDD

1. **RED:** Declared `Model::prepare_*` + `llaisys-prepare-parity-test` with oracle copies of pre-lift Qwen2 behavior. Link failed with undefined references to `Model::prepare_*`.
2. **GREEN:** Implemented helpers in `model.cpp` (Chinese comments preserved). Parity test passed element-wise vs oracle for 2-seq prefill/decode fixtures.
3. **Wrap:** `Qwen2::prepare_*` → one-line calls into `Model`.

## API

| Method | Notes |
|---|---|
| `Model::prepare_block_table(seqs, block_table_width)` | Width parameterized (Qwen2 passes `MAX_BLOCK_NUM`) |
| `Model::prepare_prefill(seqs)` | Returns `Qwen2Pack` |
| `Model::prepare_decode(seqs)` | Returns `Qwen2Pack` |

`Qwen2Pack` extracted to `src/models/qwen2_pack.hpp` so framework can return it without circular includes.

## Files

- `src/models/framework/model.hpp` / `model.cpp` — prepare implementations
- `src/models/qwen2_pack.hpp` — shared pack struct
- `src/models/qwen2.hpp` / `qwen2.cpp` — thin wrappers; include pack header
- `test/models/prepare_parity_test.cpp` — oracle parity test
- `xmake.lua` — target `llaisys-prepare-parity-test`

## Verification

```text
XMAKE_ROOT=y xmake build llaisys-core-test && xmake run llaisys-core-test
→ llaisys core test smoke assertion passed

XMAKE_ROOT=y xmake build llaisys-prepare-parity-test && xmake run llaisys-prepare-parity-test
→ prepare parity test passed
```

Confirmed: no `use_layer` / `forward_layers` / `LLAISYS_QWEN2_*` switches in `qwen2.cpp`; worker still calls existing `forward`.

## Concerns / follow-ups (Task 2+)

- `Qwen2Pack` still carries sampling fields; bridge to `BatchRuntime` + Context deferred.
- Framework `Model` now depends on `Sequence` / `Qwen2Pack` (acceptable for this lift).
- Parity test uses an embedded oracle, not a live `Qwen2` instance (avoids device/weight setup).
