# flash_decoding Integration Design

**Date:** 2026-07-28
**Status:** approved
**Scope:** Wire `is_prefill` parameter through C API / Python binding layers; simplify op.cpp 1D/2D branching.

## Motivation

`flash_decoding` is a new decode-phase attention kernel implemented under
`src/ops/paged_attention/nvidia/flash_decoding_nvidia.cu`. The existing
`paged_attention` op entry point (`op.hpp` / `op.cpp`) already accepts a
`bool is_prefill` parameter that routes to `nvidia::flash_decoding()` when
`is_prefill=false`. However, the C API, C→C++ bridge, and Python bindings
were not updated with this parameter, causing compilation failure.

Additionally, the `op.cpp` contains 1D block_ids handling branches that
are never used — all callers pass 2D batched tensors. These branches should
be removed to reduce unnecessary complexity.

## Design

### 1. Simplify op.cpp — remove 1D block_ids branching

`src/ops/paged_attention/op.cpp` currently checks `block_ids->ndim()` to
support both 1D (per-sample) and 2D (batched) forms. In practice, the
scheduler always produces batched tensors, making the 1D path dead code.

Changes:
- `batch_size = block_ids_shape[0]` (always)
- `max_block_num = block_ids_shape[1]` (always)
- `cut_idx` assert: always `batch_size + 1`
- `tot_len` assert: always `batch_size`
- Remove `if (block_ids->ndim() == 1)` / `else` branching around
  `batch_size` / `max_block_num` derivation and around cut_idx / tot_len
  validation

The device dispatch, PROFILE_STEP, static intermediate tensors, and actual
kernel calls remain unchanged.

### 2. Add `is_prefill` to C API chain

| File | Change |
|---|---|
| `include/llaisys/ops.h:16` | `llaisysPagedAttention` signature: add `bool is_prefill` |
| `src/llaisys/ops.cc:66-85` | Bridge function: accept and forward `is_prefill` |

### 3. Add `is_prefill` to Python bindings

| File | Change |
|---|---|
| `python/llaisys/libllaisys/ops.py` | Import `c_bool`; add `c_bool` to `llaisysPagedAttention.argtypes` |
| `python/llaisys/ops.py:62-81` | Fix wrapper (inner function was defined but never called); add `is_prefill: bool = True` parameter, defaulting to prefill for backward compatibility |

### 4. Update tests

`test/ops/paged_attention.py`:
- Legacy 1D calling convention (`block_ids_len, totlen, scale` as scalar ints)
  was already broken (passed ints where C API expects `llaisysTensor_t`); the
  wrapper was a no-op (inner function defined but never called).
- Remove the legacy benchmark path. All call sites use the batched form:
  `(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, max_seq_len, scale)`.

### Files NOT modified

- `src/ops/paged_attention/op.hpp` — already has `is_prefill`
- `src/ops/paged_attention/nvidia/flash_decoding_nvidia.cuh` / `.cu` — kernel code
- `src/profiler/profiler.hpp` / `.cpp` — profile system (prefill/decode distinguished
  by `snapshot.phase`, not operator name)
- `src/models/qwen2.cpp` — model forward pass and profile registration
- `xmake.lua` / `xmake/nvidia.lua` — glob `*.cu` already covers flash_decoding
