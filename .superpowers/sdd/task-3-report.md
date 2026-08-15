# Task 3 Report: Qwen2 Attention Sample + NVIDIA Capture Check

## Status

PASS

## Commit

`e96f813 feat: add qwen2 attention sample and nvidia capture check`

## Implementation

- Added `Qwen2Attention`, which stores its `const AttnWeights *` at construction and uses only that pointer during `forward`.
- Extended `ModelContext` with attention workspace tensors, per-layer `k_caches` / `v_caches`, and optional layer-zero capture tensors.
- Added `run_qwen2_layer_stack`, matching the production order: embedding, attention, FFN, last-token extraction, output norm, and LM-head projection.
- Added `llaisys-model-layer-nvidia-test`. Its `legacy_qwen2_forward_copy` cites and copies the current production sequence from `src/models/qwen2.cpp:405-466`.

## Tests

1. RED: `xmake f --nv-gpu=y -cv && xmake build llaisys-model-layer-nvidia-test`
   failed as expected because `stack.hpp` did not yet exist.
2. `xmake build llaisys-model-layer-nvidia-test && xmake run llaisys-model-layer-nvidia-test`
   passed: `qwen2 NVIDIA layer stack capture test passed`.
3. `git diff --name-only -- src/models/qwen2.cpp src/models/qwen2.hpp`
   produced no output.
4. `git diff --check`
   produced no output.

## Concerns

- `clang-format` is not installed in this environment, so formatting could not be run.
- Xmake emits an existing CUDA-flag warning: `add_cuflags("-Wno-unknown-pragmas") is ignored`.

## Important Finding Fixes

- Passed `attn_acc`, `attn_sum`, and `attn_max` through `Qwen2Attention::forward` into `paged_attention`, matching the production call site's flash-decoding contract.
- Made layer-zero attention and FFN captures independent D2D snapshots. The NVIDIA fixture uses two layers and asserts that stack and legacy golden captures do not alias `x`, `x_attn`, or `x_mlp`.
- Added a decode (`is_prefill=false`) stack invocation with correctly sized F32 flash-decoding workspaces; it would fail at the flash-decoding buffer assertion without the forwarding fix.

## Important Finding Verification

1. `xmake f --nv-gpu=y -cv && xmake build llaisys-model-layer-nvidia-test && xmake run llaisys-model-layer-nvidia-test`
   passed: `qwen2 NVIDIA layer stack capture test passed`.
2. `xmake f --nv-gpu=n -cv && xmake build llaisys-model-layer-cpu-test && xmake run llaisys-model-layer-cpu-test`
   passed: `qwen2 embedding and ffn CPU layer tests passed`.

---

## Follow-up (2026-08-15): File structure + capture/decode fix

### Status

PASS

### Commits

`bdc536b`

### Changes

1. **File structure**
   - Deleted standalone `weight_set.hpp` / `weight_set.cpp`.
   - Moved `set_weight` onto `Model` (`model.hpp` + new `model.cpp` with prepare_* TODO stubs).
   - Kept `Layer` interface-only; Embedding / Attention / FFN remain separate hpp+cpp with Chinese step comments + local aliases.
   - `xmake.lua`: framework glob comment notes `model.cpp`; core/cpu tests link cublas when `--nv-gpu=y`.

2. **Capture / decode fix**
   - Stack + legacy snapshots use CPU `stable_capture` (`tensor->to(LLAISYS_DEVICE_CPU, 0)`) after attn/ffn residual swap.
   - Decode NVIDIA test prefills the same context/caches first, then runs `is_prefill=false` decode; NaNs rejected (`isfinite` check).
   - Prefill-only compare remains a separate test.
   - Production `qwen2.cpp` / `qwen2.hpp` / python untouched.

### Tests

```bash
xmake f --nv-gpu=y -y
xmake build llaisys-core-test && xmake run llaisys-core-test
xmake build llaisys-model-layer-cpu-test && xmake run llaisys-model-layer-cpu-test
xmake build llaisys-model-layer-nvidia-test && xmake run llaisys-model-layer-nvidia-test
```

All passed. `git diff --name-only -- src/models/qwen2.cpp src/models/qwen2.hpp` empty.

### Concerns

- Core-test still prints the expected zero-token Sequence rejection to stderr (`token_ids must not be null`); assertion still passes.
- `clang-format` still unavailable in this environment.
