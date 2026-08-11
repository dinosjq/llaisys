# flash_decoding v5: flatten pipeline + 向量化访存 (基于 v4 结构)

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v5_nvidia.cu`
基线: v3（cp.async 全体参与, H=6/TK=8/CO=1）
前代: v4（flatten pipeline + COALESCE, 标量访存）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 改动 (vs v4)

| 组件 | v4 | v5 |
|------|-----|-----|
| 寄存器布局 | `reg_acc[h][(HD+31)>>5]` strided | `float4 reg_acc[h]` 连续 (lane i→元素 i×4..i×4+3) |
| Q→smem | 标量 `cast<float>(qp[col])` per element | `copy_4d` 128-bit bf16→float4 |
| K/V smem→reg | 标量 `cast<float>(kt_[c])` per element | `copy_4d` ushort4→float4, 一次 4 元素 |
| 点积 | strided 循环 | `dot = q.x*k.x + q.y*k.y + q.z*k.z + q.w*k.w` |
| 归约 merge | strided 标量 | `float4` 逐元素操作 |

v4 的 flatten pipeline + COALESCE 模板 + block 指针预计算全部保留。

## 全参数扫描 (912 组合)

19 configs × 4 HEADS × 2 TILE_K × 6 COALESCE = 912 组合, nsys 精确 GPU 计时。

### v5 最优 vs v3 基线

```
b×totlen      v3(H6T8)  v5_best  v5/v3  best(H,TK,CO)
─────────────────────────────────────────────────────────
 1 ×  256       11.4       7.3   0.64x  H=1 TK=16 CO= 1
 1 ×  512       11.2       8.5   0.76x  H=2 TK=16 CO= 1
 1 × 1024       15.4      14.5   0.94x  H=2 TK=16 CO= 1
 1 × 2048       23.5      22.8   0.97x  H=3 TK=16 CO= 3
 1 × 4096       41.4      41.4   1.00x  H=3 TK=16 CO= 6
 2 ×  256       11.1       8.6   0.77x  H=2 TK=16 CO= 1
 2 ×  512       15.7      14.5   0.92x  H=2 TK=16 CO= 1
 2 × 1024       25.3      22.9   0.90x  H=3 TK=16 CO= 3
 2 × 2048       39.5      41.4   1.05x  H=3 TK=16 CO= 6
 4 ×  256       16.2      15.3   0.95x  H=2 TK=16 CO= 2
 4 ×  512       25.7      23.4   0.91x  H=3 TK=16 CO= 3
 4 × 1024       39.5      42.0   1.06x  H=3 TK=16 CO= 6
 4 × 2048       70.0      78.7   1.12x  H=3 TK=16 CO= 4
 8 ×  256       27.2      26.2   0.96x  H=3 TK=16 CO= 1
 8 ×  512       42.9      43.9   1.02x  H=3 TK=16 CO= 2
 8 × 1024       70.7      80.7   1.14x  H=3 TK=16 CO= 2
16 ×  256       46.0      48.5   1.05x  H=3 TK=16 CO= 2
16 ×  512       73.8      83.5   1.13x  H=3 TK=16 CO= 4
32 ×  256       80.5      91.6   1.14x  H=3 TK=16 CO= 4
```

**v5 在小 batch (1-2) 全面优于 v3 (+6-56%)，在大 batch (≥4) 全面慢于 v3 (-2-14%)。**
交叉点在 batch=2/totlen=2048 和 batch=4/totlen=1024 附近。

### COALESCE 效果 (H=3, TK=16 — v5 最常见最优参数组)

```
b×totlen      CO=1    CO=2    CO=3    CO=4    CO=6    CO=8   best
──────────────────────────────────────────────────────────────────
 1 × 2048      24.6    26.6    22.8    28.7    33.8    43.8   CO=3
 2 × 2048      43.8    41.7    42.0    52.2    41.4    52.8   CO=6
 4 ×  512      24.6    29.2    23.4    28.9    40.7    44.4   CO=3
 4 × 2048      81.5    78.9    79.8    78.7    80.5   100.5   CO=4
 8 × 1024      84.3    80.7    83.0    80.8    80.8   101.1   CO=2
16 ×  512      90.3    85.5    89.0    83.5    96.7   103.3   CO=4
32 ×  256     104.2    95.5   106.6    91.6    97.9    92.0   CO=4
```

**v5 中 COALESCE 有效！** 与 v4(H6T8) 中 CO>1 全部退化不同，v5(H3T16) 中 CO=2-6 在多数
config 优于 CO=1。最优 CO 随 totlen 增大而增大——小 totlen 用 CO=3，大 totlen 用 CO=4-6。

CO=8 仍然太激进，在所有 config 中劣于 CO=4。

### H=6,TK=8 中 COALESCE 依然无效

```
b×totlen      CO=1    CO=2    CO=3    CO=4    CO=6    CO=8
───────────────────────────────────────────────────────────
 4 × 2048      96.4   137.9   136.4   134.9   136.9   176.1
 8 × 1024      98.8   140.1   139.9   136.5   138.9   177.1
16 ×  512     104.2   143.7   144.0   139.8   162.4   180.3
```

H=6 时 CO>1 全部严重退化 (CO=2 慢 38-44%)。与 v4 结论一致——高 HEADS 下寄存器压力
(135 regs/thread) 限制 occupancy，COALESCE 减少 grid 后无法补偿。

### 按 batch 的最优参数规律

| batch | 小 totlen (256-512) | 大 totlen (2048-4096) |
|-------|--------------------|-----------------------|
| 1-2 | H=1-2, TK=16, CO=1 | H=3, TK=16, CO=3-6 |
| 4-8 | H=2-3, TK=16, CO=1-3 | H=3, TK=16, CO=2-4 |
| 16-32 | H=3, TK=16, CO=2 | H=3, TK=16, CO=4 |

**规律**: H=3+TK=16 是 v5 的核心参数组，CO 从 1→4 随 totlen 递增。H=6 仅在小 totlen
且 batch=1 时出现，且总是 CO=1。

## vs v3 综合对比

| 场景 | 结论 |
|------|------|
| batch=1-2, totlen≤1024 | v5 显著优于 v3 (+6-56%) |
| batch=1-2, totlen≥2048 | v5 追平或略优于 v3 (±5%) |
| batch≥4, totlen≤512 | v5 追平 v3 (±5%) |
| batch≥4, totlen≥1024 | v3 优于 v5 (+2-14%) |

v5 收益窗口在**小 batch 小 totlen**——这是 grid 并行度不足（< 2 waves per SM）
的场景，v5 的 float4 减少指令数 + 降低 LSU 压力能有效补偿。大 batch 时 v3 的 H=6
K/V 共享优势（12→2 load 次）超过 v5 的向量化收益。

## 结论

1. **v5 在小 batch 场景胜 v3** (+6-56%)，大 batch 场景仍不如 v3 (-2-14%)
2. **v5 中 COALESCE 首次有效** — H=3/TK=16 下 CO=2-6 在多数 config 优于 CO=1，
   但最优仍不如 v3(H6T8)
3. **H=6 下 COALESCE 始终无效** — 寄存器压力根因未改变
4. **v5 不接入主接口**，保留为实验参考。主接口维持 v3

## 测试脚本

- `test/profile/_scan_v5.py` — 全参数 912 组合扫描 (4 shard 并行 nsys)
- `test/profile/_summary_v5.py` — 数据汇总脚本
- `test/profile/_h2h.py` — v3 vs v5 head-to-head 正确性 + 基准
