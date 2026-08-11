# flash_decoding v5: 向量化访存 + float4 寄存器实验

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v5_nvidia.cu`
基线: v3（cp.async 全体参与, strided 寄存器 + 标量 smem 访问）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 改动（vs v3）

| 组件 | v3 | v5 |
|------|----|----|
| Q→smem | 标量循环 `cast<float>(qp[col])` per element | `copy_4d` (128-bit bf16→float4) |
| 寄存器布局 | `reg_acc[h][(HD+31)>>5]` strided (lane i 存元素 i, i+32, i+64, i+96) | `float4 reg_acc[h]` 连续 (lane i 存元素 i*4..i*4+3) |
| K/V smem→reg | 标量 `cast<float>(kt_[c])` per element | `load_float4_from_T()` ushort4→float4, 一次 4 元素 |
| 点积 | `for (c=lane_id; c<HD; c+=32) dot += qh[c]*kt[c]` | `dot = q.x*k.x + q.y*k.y + q.z*k.z + q.w*k.w` |
| 归约 merge | strided 标量循环 | `float4` 逐元素操作 |

cp.async pipeline、tile 循环、reduce kernel 不变。

## H=6,TK=8（v3 默认参数）

```
batch×totlen    v3(us)    v5(us)   v5/v3
──────────────────────────────────────────
1 ×  256         11.4      12.0     0.95x  (v5 慢)
1 ×  512         11.2      11.8     0.95x
1 × 1024         15.2      20.9     0.73x
1 × 2048         23.6      30.9     0.77x
1 × 4096         41.4      58.9     0.70x
2 ×  256         11.1      11.6     0.95x
2 × 2048         39.5      57.5     0.69x
4 × 1024         39.3      57.8     0.68x
4 × 2048         69.8     104.1     0.67x
8 ×  512         39.8      58.5     0.68x
8 × 1024         70.6     105.7     0.67x
16 ×  256         41.6      61.8     0.67x
16 ×  512         74.2     110.9     0.67x
32 ×  256         79.8     119.7     0.67x
```

**H=6 下 v5 全面退化。** 小 batch 退化 5%，batch≥4 退化 27-50%。

## H=2,TK=16

```
batch×totlen    v3(us)    v5(us)   v5/v3
──────────────────────────────────────────
1 ×  256          7.3       6.5     1.11x  (v5 快)
1 ×  512          9.4       7.9     1.19x
1 × 1024         16.2      13.3     1.22x
1 × 2048         29.9      24.1     1.24x
1 × 4096         56.0      45.7     1.22x
2 ×  256          9.3       7.8     1.20x
2 × 2048         54.5      43.9     1.24x
4 × 1024         55.0      44.7     1.23x
4 × 2048        105.5      85.1     1.24x
8 ×  512         57.7      46.2     1.25x
8 × 1024        108.1      89.0     1.21x
16 ×  256         61.1      50.7     1.20x
16 ×  512        117.8      99.2     1.19x
32 ×  256        130.3     113.7     1.15x
```

**H=2 下 v5 全面加速 11-25%。** batch≥4 时稳定 1.19-1.25x。

## 综合最优 vs v3 最优

```
                  v3 最优             v5 最优
batch×totlen   (H,TK)  us         (H,TK)  us      v5/v3
─────────────────────────────────────────────────────────
1 ×  256       H2,T16   7.3      H2,T16   6.5    1.11x
1 ×  512       H2,T16   9.4      H2,T16   7.9    1.19x
1 × 1024       H6,T8   15.2      H2,T16  13.3    1.14x
1 × 2048       H6,T8   23.6      H2,T16  24.1    0.98x
4 × 2048       H6,T8   69.8      H2,T16  85.1    0.82x
8 × 1024       H6,T8   70.6      H2,T16  89.0    0.79x
16 ×  512      H6,T8   74.2      H2,T16  99.2    0.75x
```

交叉点在 batch=1, totlen=1024 附近。batch≥4 时 v5 最优(H2,T16) 仍慢于 v3 最优(H6,T8) 18-25%。

## 分析

1. **H=6 退化根因：寄存器压力。** `float4 reg_acc[6]` = 24 个 float 寄存器，加上 `reg_sum/max[6]` = 36 个。135 regs/thread 已在 v3 中限制 occupancy=1 block/SM。v5 的 float4 点积展开 + copy_4d Q-load 额外增加约 10-15 个临时寄存器，触发了 register spilling 到 local memory。

2. **H=2 加速根因：vectorized smem read 减少指令数。** H=2 时 reg_acc 仅 8 floats，寄存器充裕。`load_float4_from_T` 将 4 次标量 ld + 4 次 cast 合并为 1 次 64-bit ld + 展开转换，减少 LSU 压力。`copy_4d` Q-load 也减少了 load 指令。

3. **交叉点说明**：小 batch/小 totlen 时 grid 并行度主导（H=2 比 H=6 有 3× grid blocks）。大 batch 时 H=6 的 K/V 共享优势（12→2 load 次）超过 H=2 的向量化收益。

## 结论

向量化方向正确（H=2 场景 +11-25%），但 float4 对高 HEADS 场景（H≥6）的寄存器增量触发了 spilling。要在大 batch 场景收益，需要：(1) 减少 HEADS 模板的寄存器用量（如将部分 reg_acc 放回 smem），或 (2) 仅在 HEADS≤3 时启用 v5 路径，HEADS≥6 回退 v3。

## 测试脚本

- `test/profile/_test_v5.py` — v5 vs v3 正确性验证
- `test/profile/_bench_v5.py` — v5 vs v3 nsys benchmark
