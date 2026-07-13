# fp16/bf16 CUDA Native Intrinsics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace manual bit-manipulation fp16/bf16 conversion functions in `nvidia_utils.cuh` with CUDA hardware intrinsics.

**Architecture:** Single-file change in `src/utils/nvidia_utils.cuh`. Replace 4 `__device__` function bodies with CUDA native `__half2float`/`__float2half`/`__bfloat162float`/`__float2bfloat16` via `_raw` structs. All 9 kernel files and 30 call sites unchanged (go through `cast<T>()`/`load_4d()`/`save_4d()` wrappers).

**Tech Stack:** CUDA 12.4, SM_89 (Ada), C++17

## Global Constraints

- 仅修改 `src/utils/nvidia_utils.cuh`，其余文件零改动
- 使用 `__half_raw` / `__nv_bfloat16_raw` 做位级桥接（非 `static_cast`）
- 保留 `__forceinline__`
- 新增 `#include <cuda_fp16.h>` 和 `#include <cuda_bf16.h>`

## Spec Reference

`docs/superpowers/specs/2026-07-14-fp16-bf16-cuda-intrinsics-design.md`

---

### Task 1: Replace conversion functions in nvidia_utils.cuh

**Files:**
- Modify: `src/utils/nvidia_utils.cuh`

**Interfaces:**
- Consumes: `fp16_t` / `bf16_t` from `types.hpp` (unchanged)
- Produces: `nv_f16_to_f32()`, `nv_f32_to_f16()`, `nv_bf16_to_f32()`, `nv_f32_to_bf16()` — same signatures, new bodies

- [ ] **Step 1: Add CUDA fp16/bf16 headers**

Open `src/utils/nvidia_utils.cuh`. After `#include <cuda_runtime.h>`, add:

```cpp
#include <cuda_fp16.h>    // __half2float, __float2half, __half_raw
#include <cuda_bf16.h>    // __bfloat162float, __float2bfloat16, __nv_bfloat16_raw
```

- [ ] **Step 2: Replace `nv_bf16_to_f32`**

Replace (lines 27-30):

```cpp
__device__ __forceinline__ float nv_bf16_to_f32(bf16_t val) {
    uint32_t bits32 = static_cast<uint32_t>(val._v) << 16;
    return __uint_as_float(bits32);
}
```

With:

```cpp
__device__ __forceinline__ float nv_bf16_to_f32(bf16_t val) {
    __nv_bfloat16_raw raw;
    raw.x = val._v;
    return __bfloat162float(raw);
}
```

- [ ] **Step 3: Replace `nv_f32_to_bf16`**

Replace (lines 32-37):

```cpp
__device__ __forceinline__ bf16_t nv_f32_to_bf16(float val) {
    uint32_t bits32 = __float_as_uint(val);
    const uint32_t rounding_bias = 0x00007FFFu + ((bits32 >> 16) & 1u);
    uint16_t bf16_bits = static_cast<uint16_t>((bits32 + rounding_bias) >> 16);
    return bf16_t{bf16_bits};
}
```

With:

```cpp
__device__ __forceinline__ bf16_t nv_f32_to_bf16(float val) {
    __nv_bfloat16_raw raw = __float2bfloat16(val);
    return bf16_t{raw.x};
}
```

- [ ] **Step 4: Replace `nv_f16_to_f32`**

Replace (lines 39-65):

```cpp
__device__ __forceinline__ float nv_f16_to_f32(fp16_t val) {
    uint16_t h = val._v;
    uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
    int32_t exponent = (h >> 10) & 0x1F;
    uint32_t mantissa = h & 0x3FFu;

    uint32_t f32;
    if (exponent == 31) {
        f32 = sign | 0x7F800000u | (mantissa << 13);
    } else if (exponent == 0) {
        if (mantissa == 0) {
            f32 = sign;
        } else {
            exponent = -14;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                exponent--;
            }
            mantissa &= 0x3FFu;
            f32 = sign | (static_cast<uint32_t>(exponent + 127) << 23) | (mantissa << 13);
        }
    } else {
        f32 = sign | (static_cast<uint32_t>(exponent + 127 - 15) << 23) | (mantissa << 13);
    }

    return __uint_as_float(f32);
}
```

With:

```cpp
__device__ __forceinline__ float nv_f16_to_f32(fp16_t val) {
    __half_raw raw;
    raw.x = val._v;
    return __half2float(raw);
}
```

- [ ] **Step 5: Replace `nv_f32_to_f16`**

Replace (lines 67-87):

```cpp
__device__ __forceinline__ fp16_t nv_f32_to_f16(float val) {
    uint32_t f32 = __float_as_uint(val);
    uint16_t sign = static_cast<uint16_t>((f32 >> 16) & 0x8000u);
    int32_t exponent = static_cast<int32_t>((f32 >> 23) & 0xFF) - 127;
    uint32_t mantissa = f32 & 0x7FFFFFu;

    if (exponent >= 16) {
        if (exponent == 128 && mantissa != 0) {
            return fp16_t{static_cast<uint16_t>(sign | 0x7E00u)};
        }
        return fp16_t{static_cast<uint16_t>(sign | 0x7C00u)};
    } else if (exponent >= -14) {
        return fp16_t{static_cast<uint16_t>(sign | ((exponent + 15) << 10) | (mantissa >> 13))};
    } else if (exponent >= -24) {
        mantissa |= 0x800000u;
        mantissa >>= (-14 - exponent);
        return fp16_t{static_cast<uint16_t>(sign | (mantissa >> 13))};
    } else {
        return fp16_t{sign};
    }
}
```

With:

```cpp
__device__ __forceinline__ fp16_t nv_f32_to_f16(float val) {
    __half_raw raw = __float2half(val);
    return fp16_t{raw.x};
}
```

- [ ] **Step 6: Build**

```bash
xmake f --nv-gpu=y -c && xmake && xmake install
```

Expected: 编译成功，无警告。

- [ ] **Step 7: Run operator correctness tests**

```bash
python test/test_ops.py --device nvidia
```

Expected: All tests PASS.

- [ ] **Step 8: Run a quick model inference smoke test**

```bash
python test/test_infer.py --model <model_path> --test --device nvidia
```

Expected: inference correctness PASS.

- [ ] **Step 9: Commit**

```bash
git add src/utils/nvidia_utils.cuh
git commit -m "perf: replace manual fp16/bf16 conversion with CUDA native intrinsics"
```
