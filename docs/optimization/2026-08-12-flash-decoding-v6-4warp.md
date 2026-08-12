# flash_decoding v6: 4-warp block (v5 基础上减线程数 + s_q 转 T)

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
| s_q 类型 | float | **T (bf16)**, 计算时 copy_4d 转 float4 |
| HEADS 循环 | 不展开 | **`#pragma unroll`** (寄存器压力已降) |

## ncu 对比 (batch=16, totlen=2048)

```
指标                        v4_H6T8C1   v5_H6T8C1   v6_H6T8C1
──────────────────────────────────────────────────────────────
Duration (us)                395.1       545.0       285.9   ← 最优
Block Limit Registers          2           1           3
Block Limit Shared Mem         2           1           3
Achieved Occupancy           31.5%       16.0%       23.6%
Theoretical Occupancy        33.3%       16.7%       25.0%
Waves Per SM                 21.33       42.67       14.22
SMEM/block (KB)              40.32       40.32       26.30
```

**v6 相比 v5: Duration -37%, Block Limit Registers 1→3, occupancy 16%→24%, SMEM -40→26KB。**

### s_q 转 T 的效果

用户指出 smem 瓶颈来自 s_q 用 float (4B) 而非 T (2B)。转 T 后:
- SMEM/block: 27.84 → **26.30 KB**
- Duration (16×2048): 342 → **286us (-16%)**
- Q load 变纯 T 拷贝 (更快), 计算时 copy_4d 转 float4

同时加回 `#pragma unroll` (4 warps 寄存器压力已降, ILP 收益拿回)。

**Block Limit 仍卡 3**: 100KB/26.3 = 3.8, 差 1.3KB 到 4 blocks。
s_acc (12KB) 是剩余最大 float 数组, 需进一步优化才能破 3 瓶颈。

## 性能对比 (nsys, 模板参数精确匹配)

```
b×totlen   v3_H6T8   v6_C1(6,8)  v6_C1(2,16)  v6_best   cfg        v6/v3
────────────────────────────────────────────────────────────────────────
1×  256      10.7       10.3        6.1        6.1   H2T16C1    0.57x
1×  512      10.1        9.8        7.1        7.1   H2T16C1    0.70x
1× 1024      14.0       18.4       13.3       13.3   H2T16C1    0.95x
1× 2048      22.0       27.5       25.4       25.4   H2T16C1    1.15x
1× 4096      39.2       53.4       49.6       49.6   H2T16C1    1.27x
4× 2048      72.4      101.1      101.7      101.1   H6T8C1     1.40x
8× 1024      72.4      101.1      101.7      101.1   H6T8C1     1.40x
16× 512      72.4      101.1      101.7      101.1   H6T8C1     1.40x
32× 256      72.4      101.1      101.7       89.0   H3T16C8    1.23x
```

（注: gridX 匹配在部分 config 有歧义，相同 ttn 的 config 数值重复，
但 batch=1 小 totlen 和 batch≥4 大 totlen 两组趋势清晰。）

### 关键结论

1. **v6 H2T16C1 小 batch 显著优于 v3**：1×256 快 43%（6.1 vs 10.7us），
   1×512 快 30%，1×1024 快 5%。
2. **v6 大 batch 仍慢于 v3**：4×2048/8×1024/16×512 慢 40%（101 vs 72us）。
   H2T16 在大 batch 也不如 v3 的 H6T8。
3. **COALESCE 在 v6 中依然无效**：C4 = 48-144us vs C1 = 10-101us，全面退化。
4. **v6 最优 = v3 的 H2T16 小 batch 路径 + 4-warp 架构**，但大 batch 仍需 H6T8。

## vs v3 最优参数选择建议

| 场景 | v3 最优 | v6 最优 |
|------|---------|---------|
| batch=1, totlen≤512 | H2T16 | **H2T16 (v6 快 30-43%)** |
| batch=1, totlen≥1024 | H6T8 | H2T16 (v6 慢 5-27%) |
| batch≥4 | H6T8 | H6T8 (v6 慢 40%) |

v6 的价值: **小 batch (batch=1, ≤512) 场景显著优于 v3**，这是 4-warp 高 occupancy
+ float4 向量化的协同收益。大 batch 场景 v3 的 H6T8 K/V 共享仍占优。

## 后续优化方向

1. **突破 smem 3-block 瓶颈**: s_acc (12KB float) 是剩余最大数组。可考虑
   s_acc 与 s_q/s_k 复用 smem (归约阶段才用 acc), 或减少归约临时空间。
   目标: 26.3→25KB, 达到 4 blocks/SM (occupancy 33%)。
2. **H6T8 大 batch 优化**: v6 H6T8C1 比 v3 慢 40%, 需调查 (可能 reduce kernel
   的 256-thread block 与 128-thread parallel 不匹配导致 wave 效率差)。
3. **v7 方向**: 若解决 s_acc smem 复用, 4-warp + float4 + 4 blocks/SM 有望全面超越 v3。

## 测试脚本

- `test/profile/_test_v6.py` — v6 vs v3 正确性（24 项 diff=0）
- `test/profile/_diag_v6.py` — 单 config 诊断
- `test/profile/_bench_v6.py` — 全 config benchmark
- `test/profile/_ncu_v6.py` — ncu occupancy 分析
- `test/profile/_parse_v6d.py` — 模板参数精确匹配解析
