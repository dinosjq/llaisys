# flash_decoding Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `is_prefill` parameter through C API / Python binding layers and simplify op.cpp 1D/2D branching.

**Architecture:** flash_decoding lives under `paged_attention/` as an internal dispatch target. The op.cpp entry point uses `is_prefill` to route between `nvidia::paged_attention()` (prefill) and `nvidia::flash_decoding()` (decode). This plan adds the parameter to the 3 outer layers (C API, bridge, Python) and cleans up dead 1D code.

**Tech Stack:** C++17, CUDA, Python ctypes

## Global Constraints

- Do NOT modify operator kernel code (`flash_decoding_nvidia.cuh` / `.cu`, `paged_attention_nvidia.cuh` / `.cu`)
- Do NOT modify `op.hpp` (already has `is_prefill`)
- Do NOT modify profile system (`profiler.hpp` / `.cpp`)
- Do NOT modify model code (`qwen2.cpp`)
- Do NOT modify build system (`xmake.lua`, `xmake/nvidia.lua`)
- Existing operator tests should pass after changes

---

### Task 1: Simplify op.cpp — remove 1D block_ids branching

**Files:**
- Modify: `src/ops/paged_attention/op.cpp:28,51-59,70-76`

**Interfaces:**
- Produces: `batch_size = block_ids_shape[0]` (always), `max_block_num = block_ids_shape[1]` (always)
- cut_idx assertion: always expects `batch_size + 1`
- tot_len assertion: always expects `batch_size`
- block_ids ndim: only accepts 2D

- [ ] **Step 1: Constrain block_ids ndim assertion to 2D only**

Replace line 28:
```cpp
ASSERT(block_ids->ndim() == 1 || block_ids->ndim() == 2, "paged_attention: block_ids must be 1D (per-sample) or 2D (batch).");
```
With:
```cpp
ASSERT(block_ids->ndim() == 2, "paged_attention: block_ids must be 2D (batch, max_block_num).");
```

- [ ] **Step 2: Simplify batch_size / max_block_num derivation**

Replace lines 51-59:
```cpp
    size_t batch_size;
    size_t max_block_num;
    if (block_ids->ndim() == 1) {
        batch_size = 1;
        max_block_num = block_ids_shape[0];
    } else {
        batch_size = block_ids_shape[0];
        max_block_num = block_ids_shape[1];
    }
```
With:
```cpp
    const size_t batch_size = block_ids_shape[0];
    const size_t max_block_num = block_ids_shape[1];
```

- [ ] **Step 3: Simplify cut_idx / tot_len validation**

Replace lines 70-76:
```cpp
    // validate cut_idx and tot_len shapes according to batch_size
    if (batch_size == 1) {
        ASSERT(cut_idx->ndim() == 1 && cut_idx->shape()[0] >= 2, "paged_attention: cut_idx must be 1D with length>=2 for single sample.");
        ASSERT(tot_len->ndim() == 1 && tot_len->shape()[0] == 1, "paged_attention: tot_len must be 1D with length 1 for single sample.");
    } else {
        ASSERT(cut_idx->ndim() == 1 && cut_idx->shape()[0] == batch_size + 1, "paged_attention: cut_idx must be 1D with length batch+1 for batched call.");
        ASSERT(tot_len->ndim() == 1 && tot_len->shape()[0] == batch_size, "paged_attention: tot_len must be 1D with length batch for batched call.");
    }
```
With:
```cpp
    ASSERT(cut_idx->ndim() == 1 && cut_idx->shape()[0] == batch_size + 1, "paged_attention: cut_idx must be 1D with length batch+1.");
    ASSERT(tot_len->ndim() == 1 && tot_len->shape()[0] == batch_size, "paged_attention: tot_len must be 1D with length batch.");
```

- [ ] **Step 4: Commit**

```bash
git add src/ops/paged_attention/op.cpp
git commit -m "refactor: simplify paged_attention op.cpp — remove 1D block_ids branching

All callers use the 2D batched form; the 1D per-sample path was dead code.
batch_size and max_block_num are now always derived from block_ids_shape[0]
and block_ids_shape[1] respectively.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: Add `is_prefill` to C API header

**Files:**
- Modify: `include/llaisys/ops.h:16`

**Interfaces:**
- Produces: `void llaisysPagedAttention(llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, size_t, float, bool)`

- [ ] **Step 1: Add `bool is_prefill` parameter to C API declaration**

Replace line 16:
```c
    __export void llaisysPagedAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k_cache, llaisysTensor_t v_cache, llaisysTensor_t block_ids_tensor, llaisysTensor_t cut_idx_tensor, llaisysTensor_t tot_len_tensor, size_t max_seq_len, float scale);
```
With:
```c
    __export void llaisysPagedAttention(llaisysTensor_t attn_val, llaisysTensor_t q, llaisysTensor_t k_cache, llaisysTensor_t v_cache, llaisysTensor_t block_ids_tensor, llaisysTensor_t cut_idx_tensor, llaisysTensor_t tot_len_tensor, size_t max_seq_len, float scale, bool is_prefill);
```

- [ ] **Step 2: Commit**

```bash
git add include/llaisys/ops.h
git commit -m "feat: add is_prefill parameter to llaisysPagedAttention C API

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: Add `is_prefill` to C→C++ bridge

**Files:**
- Modify: `src/llaisys/ops.cc:66-85`

**Interfaces:**
- Consumes: `llaisysPagedAttention` C API signature from Task 2
- Produces: Forward `is_prefill` to `llaisys::ops::paged_attention()`

- [ ] **Step 1: Add `bool is_prefill` to bridge function signature and forward it**

Replace lines 66-85:
```cpp
void llaisysPagedAttention(llaisysTensor_t attn_val,
                           llaisysTensor_t q,
                           llaisysTensor_t k_cache,
                           llaisysTensor_t v_cache,
                           llaisysTensor_t block_ids_tensor,
                           llaisysTensor_t cut_idx_tensor,
                           llaisysTensor_t tot_len_tensor,
                           size_t max_seq_len,
                           float scale) 
{
    llaisys::ops::paged_attention(attn_val->tensor,
                                   q->tensor,
                                   k_cache->tensor,
                                   v_cache->tensor,
                                   block_ids_tensor->tensor,
                                   cut_idx_tensor->tensor,
                                   tot_len_tensor->tensor,
                                   max_seq_len,
                                   scale);
}
```
With:
```cpp
void llaisysPagedAttention(llaisysTensor_t attn_val,
                           llaisysTensor_t q,
                           llaisysTensor_t k_cache,
                           llaisysTensor_t v_cache,
                           llaisysTensor_t block_ids_tensor,
                           llaisysTensor_t cut_idx_tensor,
                           llaisysTensor_t tot_len_tensor,
                           size_t max_seq_len,
                           float scale,
                           bool is_prefill)
{
    llaisys::ops::paged_attention(attn_val->tensor,
                                   q->tensor,
                                   k_cache->tensor,
                                   v_cache->tensor,
                                   block_ids_tensor->tensor,
                                   cut_idx_tensor->tensor,
                                   tot_len_tensor->tensor,
                                   max_seq_len,
                                   scale,
                                   is_prefill);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/llaisys/ops.cc
git commit -m "feat: forward is_prefill in llaisysPagedAttention C→C++ bridge

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: Build verification (C++ side)

**Files:**
- (verification only, no code changes)

- [ ] **Step 1: Build to verify compilation**

```bash
cd /home/songjq/llaisys-main && xmake 2>&1
```
Expected: clean build, no errors.

- [ ] **Step 2: Commit** (skip if build is clean — no changes to commit)

If the build passes, proceed to Task 5. If not, fix issues and re-verify.

---

### Task 5: Add `is_prefill` to Python ctypes binding

**Files:**
- Modify: `python/llaisys/libllaisys/ops.py:1-2,39-50`

**Interfaces:**
- Consumes: `llaisysPagedAttention` C API signature from Task 2
- Produces: ctypes argtypes with `c_bool` for `is_prefill`

- [ ] **Step 1: Import `c_bool`**

Replace line 2:
```python
from ctypes import c_float, c_size_t
```
With:
```python
from ctypes import c_bool, c_float, c_size_t
```

- [ ] **Step 2: Add `c_bool` to argtypes**

Replace lines 39-50:
```python
    lib.llaisysPagedAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k_cache
        llaisysTensor_t,  # v_cache
        llaisysTensor_t,  # block_ids
        llaisysTensor_t,  # cut_idx
        llaisysTensor_t,  # tot_len
        c_size_t,         # max_seq_len
        c_float,          # scale
    ]
    lib.llaisysPagedAttention.restype = None
```
With:
```python
    lib.llaisysPagedAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k_cache
        llaisysTensor_t,  # v_cache
        llaisysTensor_t,  # block_ids
        llaisysTensor_t,  # cut_idx
        llaisysTensor_t,  # tot_len
        c_size_t,         # max_seq_len
        c_float,          # scale
        c_bool,           # is_prefill
    ]
    lib.llaisysPagedAttention.restype = None
```

- [ ] **Step 3: Commit**

```bash
git add python/llaisys/libllaisys/ops.py
git commit -m "feat: add c_bool is_prefill to llaisysPagedAttention ctypes binding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 6: Fix Python Ops wrapper

**Files:**
- Modify: `python/llaisys/ops.py:1-3,62-81`

**Interfaces:**
- Consumes: `LIB_LLAISYS.llaisysPagedAttention` ctypes binding from Task 5
- Produces: `Ops.paged_attention()` with working dispatch, `is_prefill: bool = True` parameter

**Current bug:** Inner function defined but never called (lines 70-81). The outer function is a no-op.

**Fix approach:** Detect calling convention by inspecting `*rest[0]` type — if it's a `Tensor`, use batched form; otherwise legacy. Default `is_prefill = True` for backward compatibility.

- [ ] **Step 1: Add `c_bool` import**

Replace line 3:
```python
from ctypes import c_float, c_size_t
```
With:
```python
from ctypes import c_bool, c_float, c_size_t
```

- [ ] **Step 2: Rewrite `paged_attention` wrapper**

Replace lines 62-81:
```python
    @staticmethod
    def paged_attention(attn_val: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_ids, *rest):
        """
        Support two call signatures:
        - per-sample (legacy): (attn_val, q, k_cache, v_cache, block_ids_tensor, block_ids_len:int, tot_len:int, scale:float)
        - batched single-shot: (attn_val, q_cat, k_cache, v_cache, block_ids_ll, cut_idx_ll, totlen_ll, max_seq_len:int, scale:float)

        The wrapper detects which form is used and adapts to call the C binding.
        """
        def paged_attention(attn_val: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_ids: Tensor, cut_idx: Tensor, tot_len: Tensor, max_seq_len: int, scale: float):
            LIB_LLAISYS.llaisysPagedAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k_cache.lib_tensor(),
                v_cache.lib_tensor(),
                block_ids.lib_tensor(),
                cut_idx.lib_tensor(),
                tot_len.lib_tensor(),
                c_size_t(max_seq_len),
                c_float(scale),
            )
```
With:
```python
    @staticmethod
    def paged_attention(attn_val: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_ids, *rest):
        """
        Batched call: (attn_val, q_cat, k_cache, v_cache, block_ids_ll, cut_idx_ll,
                        totlen_ll, max_seq_len:int, scale:float[, is_prefill:bool])

        Defaults is_prefill=True for backward compatibility.
        """
        # Detect calling convention: if rest[0] is a Tensor, it's the batched form
        if len(rest) >= 3 and isinstance(rest[0], Tensor):
            cut_idx = rest[0]
            tot_len = rest[1]
            max_seq_len = rest[2]
            scale = rest[3]
            is_prefill = rest[4] if len(rest) > 4 else True
            LIB_LLAISYS.llaisysPagedAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k_cache.lib_tensor(),
                v_cache.lib_tensor(),
                block_ids.lib_tensor(),
                cut_idx.lib_tensor(),
                tot_len.lib_tensor(),
                c_size_t(max_seq_len),
                c_float(scale),
                c_bool(is_prefill),
            )
        else:
            raise TypeError(
                "paged_attention: legacy per-sample signature is no longer supported. "
                "Use the batched form: (attn_val, q, k_cache, v_cache, block_ids, "
                "cut_idx, tot_len, max_seq_len, scale[, is_prefill])"
            )
```

- [ ] **Step 3: Commit**

```bash
git add python/llaisys/ops.py
git commit -m "fix: paged_attention Python wrapper — dispatch to C API, add is_prefill

The previous wrapper defined an inner function but never called it (no-op).
Now properly dispatches batched form to the C binding with is_prefill=True
as default for backward compatibility. Legacy per-sample form removed.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: Fix tests — 2D block_ids + remove broken legacy benchmark call

**Files:**
- Modify: `test/ops/paged_attention.py:49-59,124-129`

**Interfaces:**
- Consumes: `Ops.paged_attention()` fixed wrapper from Task 6 (batched form only)
- Produces: `build_block_ids()` returns 2D LLAISYS tensor `(1, len(values))`

Two changes needed:
1. `build_block_ids` creates 1D block_ids, but Task 1 simplified op.cpp to require 2D `(batch, max_block_num)`. For single-sample tests, that's `(1, len(values))`.
2. The legacy benchmark call at line 127 passes scalar ints where the C API expects tensor handles — was broken, replace with batched form.

- [ ] **Step 1: Update `build_block_ids` to create 2D block_ids**

Replace lines 49-59:
```python
def build_block_ids(values, device_name, device_id=0):
    torch_device = torch.device(f"cuda:{device_id}") if device_name == "nvidia" else torch.device("cpu")
    torch_ids = torch.tensor(values, dtype=torch.int32, device=torch_device)
    block_ids = llaisys.Tensor(
        (len(values),),
        dtype=llaisys.DataType.I32,
        device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
        device_id=device_id,
    )
    block_ids.load(c_void_p(torch_ids.data_ptr()))
    return torch_ids, block_ids
```
With:
```python
def build_block_ids(values, device_name, device_id=0):
    torch_device = torch.device(f"cuda:{device_id}") if device_name == "nvidia" else torch.device("cpu")
    torch_ids = torch.tensor(values, dtype=torch.int32, device=torch_device)
    block_ids = llaisys.Tensor(
        (1, len(values)),  # 2D: (batch=1, max_block_num)
        dtype=llaisys.DataType.I32,
        device=llaisys.DeviceType.NVIDIA if device_name == "nvidia" else llaisys.DeviceType.CPU,
        device_id=device_id,
    )
    block_ids.load(c_void_p(torch_ids.data_ptr()))
    return torch_ids, block_ids
```

- [ ] **Step 2: Replace legacy benchmark call with batched form**

Replace lines 124-129:
```python
    if profile:
        benchmark(
            lambda: torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale),
            lambda: llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, len(block_ids_values), totlen, scale),
            device_name,
        )
```
With:
```python
    if profile:
        benchmark(
            lambda: torch_paged_attention(attn_val, q, k_cache, v_cache, block_ids_values, scale),
            lambda: llaisys.Ops.paged_attention(attn_val_, q_, k_cache_, v_cache_, block_ids, cut_idx_ll, totlen_ll, seqlen, scale),
            device_name,
        )
```

- [ ] **Step 3: Commit**

```bash
git add test/ops/paged_attention.py
git commit -m "fix: update paged_attention tests — 2D block_ids, remove legacy benchmark

build_block_ids now creates 2D (1, max_block_num) tensors matching the
simplified op.cpp. Legacy benchmark call (scalar ints where C API expects
tensor handles) replaced with batched form.

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 8: Full build and test verification

**Files:**
- (verification only, no code changes)

- [ ] **Step 1: Rebuild from clean**

```bash
cd /home/songjq/llaisys-main && xmake -r 2>&1
```
Expected: clean build, no errors.

- [ ] **Step 2: Run paged_attention operator tests**

```bash
cd /home/songjq/llaisys-main && python test/ops/paged_attention.py --device nvidia 2>&1
```
Expected: `Test passed!`

- [ ] **Step 3: Commit** (skip if tests pass — no changes)
```
