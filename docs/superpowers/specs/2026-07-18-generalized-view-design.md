# Generalized Tensor::view() — Stride-Compatible Shape Transform

**Status:** approved design (2026-07-18)
**Scope:** `src/tensor/tensor.cpp` + `test/test_tensor.py` only
**Dependencies:** none (pure tensor-layer enhancement)

## 1. Motivation

Current `Tensor::view()` (`src/tensor/tensor.cpp:303-320`) throws unless the input tensor is globally contiguous (`isContiguous()`). This is overly restrictive: a non-contiguous tensor can often be reshaped zero-copy when its stride layout is compatible with the new shape. PyTorch's `view()` supports this; LLAISYS should too.

**Motivating case:** a column-sliced fused QKV tensor `[tot_seq_len, 1536]` with strides `[2048, 1]` (row-gap due to interleaved k/v) should be viewable as `[tot_seq_len, 12, 128]` — splitting the contiguous inner dimension — without a copy. Today `view()` throws; `reshape()` silently copies.

This generalization is prerequisite infrastructure for operator fusion (QKV, gate/up) which produces interleaved outputs that downstream operators must consume without copies.

## 2. Design

### 2.1 Algorithm: ATen `computeStride` semantics

Walk old dimensions from innermost outward, grouping adjacent dims into
"relatively contiguous chunks", then match new dims against those chunks.

**Definition.** Adjacent old dims `d` and `d+1` belong to the same chunk iff:
```
old_shape[d] == 1  OR  old_strides[d] == old_strides[d+1] * old_shape[d+1]
```
(i.e. a size-1 old dim never forces a chunk boundary, regardless of its stride.)

**Algorithm:**

```
view_numel = 1    // product of new dims consumed so far (working inward)
chunk_base = 0    // stride of the innermost dim in the current chunk

for each old dim d from innermost to outermost:
    view_numel *= old_shape[d]
    if old_shape[d] == 1:
        continue  // skip — no boundary, no chunk contribution
    if d == innermost OR old_strides[d] == old_strides[d+1] * old_shape[d+1]:
        // still in the same chunk
        chunk_base = old_strides[d]  // track the innermost-normal stride
        continue
    else:
        // chunk boundary — consume new dims for this chunk
        view_numel, chunk_base = consume_chunk(chunk_numel, ...)
        chunk_numel = old_shape[d]
        chunk_base  = old_strides[d]
        // restart new chunk

consume remaining chunk (the outermost chunk)
```

**Consuming new dims within a chunk.** Iterate new shape dims from innermost outward; for each chunk, the product of consumed new dims must exactly equal the chunk's `numel`. New strides are computed from `chunk_base`:

```
new_stride = chunk_base     // stride of innermost new dim in this chunk
for each new dim j consumed (innermost to outermost within chunk):
    new_strides[j] = new_stride
    new_stride *= new_shape[j]
    view_numel *= new_shape[j]
```

**Divergences from ATen for internal consistency:**
- **numel==0:** always succeed; emit packed row-major strides (same as `Tensor::create`), NOT ATen's `max(1, ·)` formula. The old `view()` used packed strides for zero-numel contiguous tensors; using a different formula would change strides for surviving zero-dim uses → violates C2.
- **ndim==0 (scalar):** succeed iff new numel == 1; emit empty stride vector.
- **Incompatible:** throw `std::runtime_error` with shape/strides context in the message.

### 2.2 Contiguous fast-path

**Keep the existing contiguous-path code as-is.** If `isContiguous()` returns true, use the original packed-stride computation verbatim:

```cpp
if (this->isContiguous()) {
    // --- old view implementation (unchanged) ---
    size_t stride = 1;
    std::vector<ptrdiff_t> strides(shape.size());
    for (size_t i = 1; i <= shape.size(); ++i) {
        strides[shape.size() - i] = stride;
        stride *= shape[shape.size() - i];
    }
    return ...;
}
// generalized (non-contiguous) path follows...
```

This makes "existing callers unchanged" true **by construction**, not by reasoning. Only non-contiguous tensors reach the new code.

### 2.3 API semantics

| API | Current behavior | New behavior |
|-----|-----------------|-------------|
| `view(shape)` | throw on non-contiguous | throw only on incompatible strides; compatible → zero-copy view |
| `reshape(shape)` | non-contiguous → always copy | non-contiguous → try `view()`; copy only when `view()` throws |
| All other methods | unchanged | unchanged |

**reshape aliasing change:** `reshape()` on a non-contiguous-but-view-compatible tensor now returns a view sharing storage, rather than a fresh copy. Audit (section 3) confirms no caller depends on the old copy semantic.

### 2.4 Error messages

Incompatible view attempts now include diagnostics:
```
"Tensor::view: shape [TOTAL] is not compatible with current shape [OLD_SHAPE] strides [OLD_STRIDES]"
```
Contrast with the old blanket `"Tensor::view error."`.

## 3. Caller Impact Audit

Full grep of `src/`, `python/`, `include/` for `view(` and `reshape(`:

**view() callers — all contiguous, zero impact:**
- `src/models/qwen2.cpp:437-457,482-484,500,517-539` — all on `slice(0, ..., ...)` of freshly created tensors → contiguous → fast-path, identical result.
- `src/llaisys/tensor.cc:74-80` — C bridge, passes through.

**reshape() callers — all contiguous, zero impact:**
- `src/models/qwen2.cpp:164-168` — dim-0 slices of contiguous cache tensor → contiguous → `view()` path (unchanged).
- `src/models/qwen2.cpp:549-550` — same pattern.

**No Python `reshape` binding exists; no Python caller can observe the semantic change.**

## 4. Testing Plan (`test/test_tensor.py`)

All new tests use CPU tensors and validate against PyTorch behavior.

### 4.1 Existing tests (regression, must pass unchanged)

The 4 existing tests (load, view, permute, slice) remain as-is — they exercise contiguous tensors and the fast-path guarantees identical results.

### 4.2 New tests

**T1. Column slice → split (motivating case):**
```python
t = torch.arange(4096, dtype=torch.float32).reshape(2, 2048)
lt = llaisys.Tensor((2, 2048), dtype=llaisys_dtype("f32"))
lt.load(t.data_ptr())

t_slice = t[:, :1536]
lt_slice = lt.slice(1, 0, 1536)

t_view = t_slice.view(2, 12, 128)
lt_view = lt_slice.view(2, 12, 128)

# shapes, strides, data match
assert lt_view.shape() == list(t_view.shape)
assert lt_view.strides() == list(t_view.stride())
assert check_equal(lt_view, t_view)
```

**T2. Permute → merge (non-contiguous but view-compatible):**
```python
t = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).permute(2, 0, 1)
# shape [5,3,4], strides [1,20,5]  (NOT contiguous: 1≠12)

lt = llaisys.Tensor((3, 4, 5), dtype=llaisys_dtype("f32"))
lt.load(torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).data_ptr())
lt = lt.permute(2, 0, 1)

# PyTorch allows view(5,12) — strides become [1,5]
t_view = t.view(5, 12)
lt_view = lt.view(5, 12)
assert lt_view.shape() == [5, 12]
assert lt_view.strides() == [1, 5]
assert check_equal(lt_view, t_view)
```

**T3. Incompatible throws (via subprocess — see §4.3):**
```python
# Column-sliced tensor cannot flatten (tot > 1)
# Script: lt_slice = lt.slice(1, 0, 1536); lt_slice.view(lt_slice.numel())
# Non-contiguous with row-gap → flatten requires merge across chunk boundary → throw
# PyTorch: RuntimeError → matching behavior
```

**T4. Incompatible does NOT throw when tot == 1 (single row → single chunk):**
```python
t = torch.arange(2048, dtype=torch.float32).reshape(1, 2048)
lt = llaisys.Tensor((1, 2048), dtype=llaisys_dtype("f32"))
lt.load(t.data_ptr())

lt_slice = lt.slice(1, 0, 1536)  # shape [1,1536], strides [2048,1] — single row
# but strides[0]=2048, strides[1]*shape[1]=1536 → NOT equal → chunk boundary for tot>1
# For tot==1: chunk {1536} → view [1536] → valid! (tot=1, no row-gap real issue)
# Confirm PyTorch agrees:
t_slice = t[:, :1536]; t_view = t_slice.view(1536)
lt_view = lt_slice.view(1536)
assert check_equal(lt_view, t_view)
```

**T5. Size-1 dimension insertion/removal:**
```python
t = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
lt = llaisys.Tensor((2, 3, 4), dtype=llaisys_dtype("f32"))
lt.load(t.data_ptr())

# slice middle dim → non-contiguous (gap after each element)
t_slice = t[:, 1:2, :]  # shape [2,1,4]
lt_slice = lt.slice(1, 1, 2)
# size-1 dim is skippable → view(2,4) should succeed
t_view = t_slice.view(2, 4)
lt_view = lt_slice.view(2, 4)
assert check_equal(lt_view, t_view)
```

### 4.3 Throw-path testing

**The C bridge `tensorView` (src/llaisys/tensor.cc:74-80) has no try/catch.** A `std::runtime_error` propagating through `extern "C"` into ctypes → `std::terminate` kills the Python interpreter. Therefore:

- **Expected-throw tests run in a subprocess** with `subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True)` and assert `result.returncode != 0`.
- **Or:** note the throw path is exercised by the C++ model tests (all of which use contiguous tensors and never trigger it), and skip explicit negative tests for now. The `view()` throw semantic on non-contiguous tensors is pre-existing; the generalized version only changes *which* shapes trigger the throw (more succeed, fewer fail). The throw path itself is mechanically a `throw std::runtime_error(...)` — same mechanism, same C bridge issue.

**Decision:** include one subprocess-based negative test (T3) to confirm the mechanism, wrapped in a try/except marking it as informational. This is low-risk since all existing throw paths already have the same C bridge limitation.

### 4.4 Test hygiene notes

- `Tensor.load()` does flat dense memcpy — load into the base tensor before creating non-contiguous views.
- `check_equal()` in test_utils.py correctly handles positive-stride non-contiguous layouts via `torch.as_strided`. It rejects negative strides, which are verified not produced by any meta op in this codebase.
- `reshape` has no Python binding — its new fallback path is covered only transitively through view tests. Documented, not claimed as tested.

## 5. Non-Goals

- NOT adding `reshape` to Python bindings
- NOT fixing pre-existing bugs (e.g., `debug()` D2H undersized buffer for non-contiguous NVIDIA tensors)
- NOT modifying any operator, C API header, or Python binding signature
- NOT doing QKV fusion (that's a separate task using this as infrastructure)

## 6. Success Criteria

1. `test/test_tensor.py` — existing 4 tests + new tests T1-T5 all pass (CPU)
2. `python test/test_ops.py` — full operator suite no regression
3. `python test/test_infer.py --model <path> --test` — inference correctness no regression
4. `python test/test_infer_batch.py --model <path> --test` — batch inference no regression (if available)
5. `python test/test_server.py --model <path>` — 6/6 pass
6. Contiguous fast-path is provably identical to old behavior (same code path, see §2.2)

## 7. Implementation Order

1. Rewrite `Tensor::view()` in `src/tensor/tensor.cpp`:
   - Guard with `isContiguous()` fast-path (keep old code block as comment above)
   - Add numel==0 and ndim==0 special cases
   - Implement chunk-walk algorithm for non-contiguous case
2. Update `Tensor::reshape()`: replace `isContiguous()` check with generalized `view()` call → catch → fallback to `contiguous()->view()`
3. Update `test/test_tensor.py`: add T1-T5
4. Build, test, iterate
5. Whole-suite regression
