# Segfault Fix + copy_4d Rollout + flash_decoding Tests — Implementation Plan

> **For agentic workers:** Use superpowers:subagent-driven-development or superpowers:executing-plans.
> Steps use checkbox (`- [ ]`) syntax.

**Goal:** Fix CUDA segfault, roll out copy_4d to eligible operators, add flash_decoding tests.

**Tech Stack:** C++17, CUDA, Python ctypes

## Global Constraints

- Do NOT modify kernel code (all `.cu` files except the load_4d→copy_4d replacements)
- Do NOT modify `nvidia_utils.cuh` (copy_4d already added)
- Do NOT modify build system
- Existing tests must continue to pass

---

### Task 1: Move flash_decoding buffers from op.cpp static to Qwen2Tensors

**Files:**
- Modify: `src/models/qwen2.hpp:35-62` — add 3 fields to Qwen2Tensors
- Modify: `src/models/qwen2.cpp:128-161` — create tensors in constructor
- Modify: `src/models/qwen2.cpp:500` — pass tensors to paged_attention call
- Modify: `src/ops/paged_attention/op.hpp` — add 3 params to signature
- Modify: `src/ops/paged_attention/op.cpp:54-60` — remove static, use passed-in
- Modify: `include/llaisys/ops.h:16` — add 3 params to C API
- Modify: `src/llaisys/ops.cc:66-86` — bridge forward
- Modify: `python/llaisys/libllaisys/ops.py:39-51` — add 3 argtypes
- Modify: `python/llaisys/ops.py:62-84` — add 3 params to wrapper

**Interfaces:**
- Produces: `paged_attention(attn_val, q, k_cache, v_cache, block_ids, cut_idx, tot_len, max_seq_len, scale, is_prefill, attn_acc, attn_sum, attn_max)`
- CPU path: passes `nullptr` for the 3 context buffers

- [ ] **Step 1: Add 3 fields to Qwen2Tensors in qwen2.hpp**

In `struct Qwen2Tensors`, after `_top_val`:
```cpp
    tensor_t _attn_acc;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, dh) F32
    tensor_t _attn_sum;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, 1)  F32
    tensor_t _attn_max;   // flash_decoding: (BATCH_MAX_BLOCK_NUM, nh, 1)  F32
```

- [ ] **Step 2: Create tensors in Qwen2 constructor (qwen2.cpp)**

After existing `_tensors._top_val` creation (line ~159), add:
```cpp
_tensors._attn_acc = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, dh}, LLAISYS_DTYPE_F32, device, device_id);
_tensors._attn_sum = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
_tensors._attn_max = Tensor::create({BATCH_MAX_BLOCK_NUM, nh, 1}, LLAISYS_DTYPE_F32, device, device_id);
```

- [ ] **Step 3: Update paged_attention call site in forward()**

In `forward()` at `ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, dev_cut_idx, dev_tot_len, max_seq_len, scale, is_prefill)`, add the 3 buffer tensors:
```cpp
ops::paged_attention(attn_val, q_rope, k_layer, v_layer, dev_block_ids, dev_cut_idx, dev_tot_len,
                     max_seq_len, scale, is_prefill,
                     ts._attn_acc, ts._attn_sum, ts._attn_max);
```

- [ ] **Step 4: Update op.hpp signature**

Replace:
```cpp
void paged_attention(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
    tensor_t block_ids, tensor_t cnt_idx, tensor_t tot_len,
    size_t max_seq_len, float scale, bool is_prefill);
```
With:
```cpp
void paged_attention(tensor_t attn_val, tensor_t q, tensor_t k_cache, tensor_t v_cache,
    tensor_t block_ids, tensor_t cnt_idx, tensor_t tot_len,
    size_t max_seq_len, float scale, bool is_prefill,
    tensor_t attn_acc = nullptr, tensor_t attn_sum = nullptr, tensor_t attn_max = nullptr);
```

- [ ] **Step 5: Update op.cpp — remove static, use passed-in tensors**

Remove lines 54-60 (`#ifdef ENABLE_NVIDIA_API ... #endif` block with static tensors).

Then in the NVIDIA flash_decoding branch, replace `attn_acc`, `attn_sum`, `attn_max` references with the new parameters:
```cpp
return nvidia::flash_decoding(attn_val->data(), attn_acc->data(), attn_sum->data(), attn_max->data(),
                               q->data(), k_cache->data(), v_cache->data(), ...);
```

- [ ] **Step 6: Update C API (ops.h)**

Add 3 `llaisysTensor_t` params to `llaisysPagedAttention`:
```c
__export void llaisysPagedAttention(llaisysTensor_t attn_val, llaisysTensor_t q,
    llaisysTensor_t k_cache, llaisysTensor_t v_cache,
    llaisysTensor_t block_ids_tensor, llaisysTensor_t cut_idx_tensor,
    llaisysTensor_t tot_len_tensor, size_t max_seq_len, float scale, bool is_prefill,
    llaisysTensor_t attn_acc, llaisysTensor_t attn_sum, llaisysTensor_t attn_max);
```

- [ ] **Step 7: Update bridge (ops.cc)**

Add 3 params to function signature and forward to `paged_attention()`. For each, convert `llaisysTensor_t → tensor_t` (nullptr-safe: `t ? t->tensor : nullptr`).

- [ ] **Step 8: Update Python ctypes binding**

Add 3 `llaisysTensor_t` to `lib.llaisysPagedAttention.argtypes`.

- [ ] **Step 9: Update Python wrapper (ops.py)**

The `paged_attention` wrapper uses `*rest` for variable arguments. The 3 buffer tensors are appended to the `*rest` unpacking:

```python
@staticmethod
def paged_attention(attn_val, q, k_cache, v_cache, block_ids, *rest):
    # rest = (cut_idx, tot_len, max_seq_len, scale[, is_prefill,
    #         attn_acc, attn_sum, attn_max])
    if len(rest) >= 3 and isinstance(rest[0], Tensor):
        # Extract required args
        cut_idx, tot_len, max_seq_len, scale = rest[0], rest[1], rest[2], rest[3]
        is_prefill = rest[4] if len(rest) > 4 and not isinstance(rest[4], Tensor) else True
        # Extract optional buffer tensors (after is_prefill or after scale if no is_prefill)
        buf_start = 5 if (len(rest) > 4 and not isinstance(rest[4], Tensor)) else 4
        attn_acc = rest[buf_start] if len(rest) > buf_start else None
        attn_sum = rest[buf_start + 1] if len(rest) > buf_start + 1 else None
        attn_max = rest[buf_start + 2] if len(rest) > buf_start + 2 else None
        LIB_LLAISYS.llaisysPagedAttention(
            attn_val.lib_tensor(), ..., c_bool(is_prefill),
            attn_acc.lib_tensor() if attn_acc else None,
            attn_sum.lib_tensor() if attn_sum else None,
            attn_max.lib_tensor() if attn_max else None,
        )
    else:
        raise TypeError("legacy per-sample signature no longer supported.")
```

Qwen2 model always passes the 3 buffers (constructed in Task 1 Step 2). External callers via Python wrapper can omit them — C API handles nullptr gracefully (prefill path doesn't need them).

- [ ] **Step 10: Build, test, commit**

```bash
xmake && xmake install
python3 test/ops/paged_attention.py --device nvidia  # expect Test passed!, no segfault on exit
```

Note: This step verifies segfault fix only (no crash on exit). flash_decoding buffer correctness is validated by Task 3 tests.

---

### Task 2: Roll out copy_4d to eligible operators

**Files:**
- Modify: `src/ops/embedding/nvidia/embedding_nvidia.cu:26-27`
- Modify: `src/ops/self_attention/nvidia/self_attention_nvidia.cu:78-80,102-104,108-109`
- Modify: `src/ops/paged_attention/nvidia/paged_attention_nvidia.cu:88-90,118-120,124-126`
- Modify: `src/ops/paged_attention/nvidia/flash_decoding_nvidia.cu:90-92` (q load — was missed in prior pass)

Each replacement follows the same pattern — replace:
```cpp
const float4 flo4 = llaisys::utils::nvidia::load_4d(SRC + col);
llaisys::utils::nvidia::save_4d(DST + col, flo4);
```
With:
```cpp
llaisys::utils::nvidia::copy_4d(SRC + col, DST + col);
```

- [ ] **Step 1: embedding_nvidia.cu** — 1 replacement (T→T)
- [ ] **Step 2: self_attention_nvidia.cu** — 3 replacements (T→float)
- [ ] **Step 3: paged_attention_nvidia.cu** — 3 replacements (T→float, 2 blocks)
- [ ] **Step 4: flash_decoding_nvidia.cu** — 1 replacement (q load, T→float)

No type parameter changes needed — `copy_4d<SrcT, DstT>` deduces from pointer types.

- [ ] **Step 5: Build, run all affected operator tests, commit**

```bash
xmake && xmake install
python3 test/ops/embedding.py
python3 test/ops/self_attention.py --device nvidia
python3 test/ops/paged_attention.py --device nvidia
git add ... && git commit -m "..."
```

---

### Task 3: flash_decoding tests + benchmark

**Files:**
- Create: `test/ops/flash_decoding.py`

**Structure** (modeled after `test/ops/paged_attention.py`):

```python
def torch_flash_decoding(attn_val, query, k_cache, v_cache, block_ids, scale, totlen):
    """Reference: same as torch_paged_attention with totlen parameter."""
    # identical to torch_paged_attention

def test_op_flash_decoding(seqlen=1, token_num, block_num, block_ids_values, nh, nkvh, hd, ...):
    # Create tensors (seqlen=1 for decode)
    # Create flash_decoding context buffers (required when is_prefill=False)
    buf_block_num = len(block_ids_values)  # or larger fixed bound
    attn_acc = llaisys.Tensor((buf_block_num, nh, hd), dtype=llaisys.DataType.F32,
                              device=llaisys_device(device_name), device_id=device_id)
    attn_sum = llaisys.Tensor((buf_block_num, nh, 1), dtype=llaisys.DataType.F32,
                              device=llaisys_device(device_name), device_id=device_id)
    attn_max = llaisys.Tensor((buf_block_num, nh, 1), dtype=llaisys.DataType.F32,
                              device=llaisys_device(device_name), device_id=device_id)
    # Call torch reference
    # Call llaisys.Ops.paged_attention(..., is_prefill=False, attn_acc, attn_sum, attn_max)
    # check_equal

def test_op_flash_decoding_batched(batch_cases, ...):
    # Multiple sequences, each seqlen=1
    # Create ctx buffers with max_block_num across batch
    # Call with is_prefill=False + buffers

def benchmark_flash_decoding(...):
    # Compare flash_decoding (is_prefill=False) vs paged_attention (is_prefill=True)
    # throughput at varying KV cache lengths
```

**Test shapes:** seqlen=1 only (decode characteristic), varying block_ids/token_num/nh/nkvh/hd.

- [ ] **Step 1: Create test file with single-sample tests**
- [ ] **Step 2: Add batched tests**
- [ ] **Step 3: Add benchmark**
- [ ] **Step 4: Build, run, verify, commit**

```bash
xmake && xmake install
python3 test/ops/flash_decoding.py --device nvidia
git add test/ops/flash_decoding.py && git commit -m "test: add flash_decoding correctness tests + benchmark"
```
