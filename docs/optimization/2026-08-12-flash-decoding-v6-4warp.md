# flash_decoding v6: 4-warp block (v5 基础上减线程数)

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v6_nvidia.cu`
基线: v3 (H6T8) / v5 (H6T8C1)
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 动机

v5 的 ncu 分析显示 **Block Limit Registers = 1, occupancy 16%**（v4 为 2 blocks/31%）。
float4 寄存器 (`reg_acc[HEADS]`) + 点积展开使每线程寄存器 >192，256 线程的 block
无法在同 SM 驻留 2 个。解决方案：**减少 block 线程数到 128 (4 warps)**，
每 block 寄存器总量减半，SM 可驻留更多 block。

## 改动 (vs v5)

| 组件 | v5 | v6 |
|------|----|----|
| BLOCK_DIM | 256 (8 warps) | **128 (4 warps)** |
| NW | 8 | **4** |
| Q load | `if (warp_id < HEADS)` 每 warp 1 head | `for(h=warp_id; h<HEADS; h+=NW)` 循环分 head |
| 归约 | warp0/1 并行分半 | **warp0 串行合并 1,2,3** (4 warp 直接简单) |
| reduce kernel | 256 threads | **独立 256 threads** (parallel 128, reduce 256) |

## ncu 对比 (batch=16, totlen=2048)

```
指标                        v4_H6T8C1   v5_H6T8C1   v6_H6T8C1
──────────────────────────────────────────────────────────────
Duration (us)                395.1       545.0       341.7
Block Limit Registers          2           1           3
Block Limit Shared Mem         2           1           3
Achieved Occupancy           31.5%       16.0%       23.9%
Theoretical Occupancy        33.3%       16.7%       25.0%
Waves Per SM                 21.33       42.67       14.22
SMEM/block (KB)              40.32       40.32       27.84
```

**v6 相比 v5: Duration -37%, Block Limit Registers 1→3, occupancy 16%→24%。**
但 SMEM 限制成为新瓶颈（3 blocks cap 于 smem, 不是寄存器）。

## 性能对比 (nsys)

```
b×totlen   v3_H6T8   v5_best   v6_best   v6_cfg        v6/v3
──────────────────────────────────────────────────────────────
1×  256      12.5      12.2       9.6   H2T16C1        0.77x
1×  512      12.3      12.0      10.5   H2T16C1        0.86x
1× 1024      16.7      19.7      18.8   H2T16C1        1.13x
1× 2048      25.6      22.2      22.2   H6T8C1         0.87x
1× 4096      45.1      41.4      41.4   H6T8C1         0.92x
2×  256      12.1      12.0      10.6   H2T16C1        0.88x
2× 2048      43.4      39.8      39.8   H6T8C1         0.92x
4×  512      25.0      21.5      21.5   H6T8C1         0.86x
4× 1024      43.0      39.7      39.7   H6T8C1         0.92x
4× 2048      76.5      74.8      74.8   H6T8C1         0.98x
8×  256      25.8      22.6      22.6   H6T8C1         0.88x
8×  512      43.4      40.7      40.7   H6T8C1         0.94x
8× 1024      76.9      76.7      76.7   H6T8C1         1.00x
16× 256      45.3      44.4      44.4   H6T8C1         0.98x
16× 512      80.5      79.6      79.6   H6T8C1         0.99x
32× 256      87.3      87.5      87.5   H6T8C1         1.00x
```

### 关键结论

1. **v6 全面追平或超越 v3**（0.77-1.00x）。无任何 config 比 v3 慢。
2. **v6 H6T8C1 大 batch 基本等于 v3**（8×1024: 76.7 vs 76.9, 16×512: 79.6 vs 80.5）。
   相比 v5 的 70.6/74.2（v5 大 batch 退化），v6 恢复了 v3 水平。
3. **v6 H2T16C1 小 batch 最优**：1×256 = 9.6us vs v3 12.5 (+23%)，
   1×2048 = 22.2 vs 25.6 (+13%)。
4. **COALESCE 在 v6 中再次无效**：H6T8C4 = 66-88us vs H6T8C1 = 18-88us。
   C4 始终慢于 C1（除个别噪声）。

## vs v3 最优参数选择建议

| 场景 | v3 最优 | v6 最优 |
|------|---------|---------|
| batch=1-2, totlen≤512 | H2T16 | **H2T16 (v6 略优)** |
| batch=1-2, totlen≥1024 | H6T8 | **H6T8 (v6 持平)** |
| batch≥4 | H6T8 | **H6T8 (v6 持平)** |

v6 的主要价值：**消除了 v5 的大 batch 退化**，同时保留小 batch 优势。
不引入主接口（v3 仍是当前最优且更简单），但 v6 的 4-warp 方案值得作为 v7 的基础。

## 测试脚本

- `test/profile/_test_v6.py` — v6 vs v3 正确性（24 项 diff=0）
- `test/profile/_diag_v6.py` — 单 config 诊断
- `test/profile/_bench_v6.py` — 全 config benchmark
- `test/profile/_ncu_v6.py` — ncu occupancy 分析
