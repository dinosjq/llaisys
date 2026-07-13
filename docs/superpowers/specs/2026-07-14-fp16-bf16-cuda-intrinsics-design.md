# fp16/bf16 CUDA 原生转换函数替换

## 日期
2026-07-14

## 动机

当前 `src/utils/nvidia_utils.cuh` 中用手动位操作实现 fp16/bf16 ↔ float32 转换，CUDA 提供了等价的硬件指令（`__half2float`、`__float2half`、`__bfloat162float`、`__float2bfloat16`），可直接替换以获取性能提升。

## 位兼容性确认

| | 项目类型 | CUDA 类型 | 位布局 |
|---|---|---|---|
| fp16 | `fp16_t` = `struct { uint16_t _v; }` | `__half` = `unsigned short` | IEEE 754: 1s+5e+10m |
| bf16 | `bf16_t` = `struct { uint16_t _v; }` | `__nv_bfloat16` = `unsigned short` | bfloat16: 1s+8e+7m |

权重文件以 raw bytes 从磁盘加载 → `uint16_t` 存储的是 IEEE 754 / bfloat16 标准位模式 → CUDA 原生类型使用完全相同位模式。**可直接 `reinterpret_cast` 级别兼容。**

## 设计

### 改动范围

**仅修改 `src/utils/nvidia_utils.cuh`**（1 个文件），其余文件零改动。

### 头文件

新增两个 include：

```cpp
#include <cuda_fp16.h>    // __half2float, __float2half, __half_raw
#include <cuda_bf16.h>    // __bfloat162float, __float2bfloat16, __nv_bfloat16_raw
```

### 替换 4 个转换函数

旧实现（手动位操作，共 ~50 行）→ 新实现（硬件指令，共 ~20 行）：

```cpp
// ── bf16 ──
__device__ __forceinline__ float nv_bf16_to_f32(bf16_t val) {
    __nv_bfloat16_raw raw;
    raw.x = val._v;
    return __bfloat162float(raw);
}

__device__ __forceinline__ bf16_t nv_f32_to_bf16(float val) {
    __nv_bfloat16_raw raw = __float2bfloat16(val);
    return bf16_t{raw.x};
}

// ── fp16 ──
__device__ __forceinline__ float nv_f16_to_f32(fp16_t val) {
    __half_raw raw;
    raw.x = val._v;
    return __half2float(raw);
}

__device__ __forceinline__ fp16_t nv_f32_to_f16(float val) {
    __half_raw raw = __float2half(val);
    return fp16_t{raw.x};
}
```

### 关键细节：为什么用 `_raw` 结构体

`__half` 的构造函数是 `__half(float)` — 将参数当作**数值**而非位模式。`static_cast<__half>(uint16_t_val)` 会走 `uint16_t → float → __half(float)` 路径，将位模式误当数值。

`__half_raw` 是 `struct { unsigned short x; }`，通过非 explicit 构造函数隐式转换为 `__half`，保持原始位模式不变。

### 不影响的部分

- `cast<T>()` 模板 — 不动
- `load_4d()` / `save_4d()` — 不动
- `add()` — 不动
- 9 个 CUDA kernel 文件 — 零改动
- `src/utils/types.cpp`（CPU 转换函数）— 不动

## 已知差异

### fp16 f32→f16 舍入行为

| | 旧实现 | 新实现 |
|---|---|---|
| 舍入方式 | 截断（向零） | 向最近偶数（IEEE 754 默认） |
| 差异 | — | ±1 ULP |

`__float2half` 的向最近偶数舍入比截断更精确。模型推理对 ±1 ULP 舍入噪声具有鲁棒性，不影响正确性。

### 硬件要求

| 指令 | 最低架构 | 当前目标 |
|---|---|---|
| `__half2float` / `__float2half` | SM_53 (Maxwell) | SM_89 (Ada) ✅ |
| `__bfloat162float` / `__float2bfloat16` | SM_80 + CUDA 11 | SM_89 + CUDA 12.4 ✅ |

当前硬件完全满足。若未来需支持 V100/T4 (SM < 80)，需为 bf16 添加架构条件编译回退到手动实现。

## 性能预期

手动位操作需要 6~15 条整数指令，CUDA 内置函数映射到单条 F2F 硬件指令。fp16 路径提升最明显，bf16→f32 的两条指令与硬件指令基本持平。
