# Segfault Fix + copy_4d Rollout + flash_decoding Tests

**Date:** 2026-07-28
**Status:** draft
**Scope:** (1) Fix segfault by moving static CUDA tensors to model, (2) Replace load_4d+save_4d with copy_4d in appropriate operators, (3) Add flash_decoding correctness tests + benchmark.

---

## 1. Segfault Fix — Move static tensors to Qwen2Tensors

### Root Cause

`src/ops/paged_attention/op.cpp:55-59` declares three `static tensor_t`:

```cpp
static tensor_t attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dv}, LLAISYS_DTYPE_F32, ...);
static tensor_t attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, ...);
static tensor_t attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, ...);
```

On process exit, static destructors run. `tensor_t = shared_ptr<Tensor>` destructor calls CUDA memory deallocation. If CUDA runtime is already shut down, `cudaFree` → segfault.

### Fix

Move these buffers into `Qwen2Tensors` struct, managed by the model's lifecycle:

| File | Change |
|---|---|
| `src/models/qwen2.hpp` | Qwen2Tensors: add `_attn_acc`, `_attn_sum`, `_attn_max` |
| `src/models/qwen2.cpp` | Constructor: create 3 tensors; forward(): pass to paged_attention |
| `src/ops/paged_attention/op.hpp` | Signature: add 3 `tensor_t` params (context buffers for flash_decoding) |
| `src/ops/paged_attention/op.cpp` | Remove `static` block; use passed-in tensors directly |
| `include/llaisys/ops.h` | `llaisysPagedAttention`: add 3 `llaisysTensor_t` params |
| `src/llaisys/ops.cc` | Bridge: forward 3 new params |
| `python/llaisys/libllaisys/ops.py` | argtypes: add 3 `llaisysTensor_t` |
| `python/llaisys/ops.py` | Wrapper: pass 3 tensors (default: create them) |

The 3 tensors are:
- `attn_acc`: shape `(BATCH_MAX_BLOCK_NUM, nh, dh)`, dtype F32 — parallel kernel partial results
  (dh == dv for Qwen2; for models with d ≠ dv, use dv)
- `attn_sum`: shape `(BATCH_MAX_BLOCK_NUM, nh, 1)`, dtype F32 — per-block denominator
- `attn_max`: shape `(BATCH_MAX_BLOCK_NUM, nh, 1)`, dtype F32 — per-block max

### Null safety

When `is_prefill=false` (flash_decoding path), all 3 buffers must be non-null. `op.cpp` asserts:
```cpp
ASSERT(attn_acc && attn_sum && attn_max,
       "paged_attention: flash_decoding requires attn_acc/sum/max buffers.");
```

### CPU path

CPU path does not use these buffers. Pass `nullptr` for CPU calls — `op.cpp` only accesses them in the NVIDIA flash_decoding branch.

---

## 2. copy_4d Rollout

### Which operators can use copy_4d

copy_4d replaces `load_4d(src) → save_4d(dst)` when there is **no computation** between load and save:

| File | Pattern | Replacement |
|---|---|---|
| `embedding_nvidia.cu:26-27` | `load_4d(src) → save_4d(dst)` | `copy_4d<T, T>(src, dst)` |
| `self_attention_nvidia.cu:78-80` | `load_4d(q) → save_4d(s_q)` | `copy_4d<T, float>(q, s_q)` |
| `self_attention_nvidia.cu:102-104` | `load_4d(k) → save_4d(s_k)` | `copy_4d<T, float>(k, s_k)` |
| `self_attention_nvidia.cu:108-109` | `load_4d(v) → save_4d(s_v)` | `copy_4d<T, float>(v, s_v)` |
| `paged_attention_nvidia.cu:88-90` | `load_4d(q) → save_4d(s_q)` | `copy_4d<T, float>(q, s_q)` |
| `paged_attention_nvidia.cu:118-120` | `load_4d(k) → save_4d(s_k)` | `copy_4d<T, float>(k, s_k)` |
| `paged_attention_nvidia.cu:124-126` | `load_4d(v) → save_4d(s_v)` | `copy_4d<T, float>(v, s_v)` |
| `flash_decoding_nvidia.cu:90-92` | `load_4d(q) → save_4d(s_q)` | `copy_4d<T, float>(q, s_q)` |

### Which operators keep load_4d/save_4d

Operators with **computation between load and save** keep separate calls:

| File | Why |
|---|---|
| `swiglu_nvidia.cu` | load gate+up → swiglu compute → save |
| `rms_norm_nvidia.cu` | load in → norm compute → save; load weight (no paired save) |
| `rope_nvidia.cu` | load in_a+in_b → rope compute → save |
| `add_nvidia.cu` | load a+b → add → save c |
| `argmax_nvidia.cu` | load vals only (no paired save) |

---

## 3. flash_decoding Tests + Benchmark

### Correctness test (`test/ops/flash_decoding.py`)

Modeled after `test/ops/paged_attention.py`, using decode-specific shapes:

**Test shapes:** seqlen=1 (decode: one token per sequence), varying KV cache lengths:

| seqlen | token_num | block_num | block_ids | nh | nkvh | hd |
|---|---|---|---|---|---|---|
| 1 | 4 | 5 | [4,1,3] | 4 | 2 | 8 |
| 1 | 8 | 10 | [9,3,7,1] | 8 | 2 | 16 |
| 1 | 4 | 12 | [11,6,2,9,0] | 12 | 3 | 8 |
| 1 | 4 | 6 | [5,2,4,0] | 4 | 1 | 8 |

**Batched test:** Multiple sequences (batch=4), each seqlen=1, varying KV cache lengths.

**Reference:** PyTorch implementation (reuse `torch_paged_attention` from paged_attention test, with totlen parameter).

**Validation:** `check_equal` with dtype-appropriate atol/rtol.

### Benchmark

Compare `flash_decoding` (is_prefill=false) vs `paged_attention` (is_prefill=true) on decode-shaped inputs, measuring throughput at different KV cache lengths.

### Files NOT modified

- `flash_decoding_nvidia.cuh` / `.cu` — kernel code
- `paged_attention_nvidia.cuh` / `.cu` — kernel code
- `xmake/nvidia.lua` — build system
- `profiler.hpp` / `.cpp` — profile system
