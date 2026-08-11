# flash_decoding v4 & v5 ncu 性能分析

日期: 2026-08-12
分析对象: `flash_decoding_v4_kernel` / `flash_decoding_v5_kernel`, batch=16/totlen=2048
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）
工具: ncu --set full, launch-count 1

## 关键指标对比

```
指标                          v4_H6T8C1    v4_H6T8C4    v5_H6T8C1    v5_H3T16C4
──────────────────────────────────────────────────────────────────────────────
Duration (us)                   395.1        395.7        545.0        546.1
Elapsed Cycles                 758,515      759,655     1,046,232    1,048,442
Grid Size                      1024          1024         1024         1024
Block Size                      256           256          256          256
──────────────────────────────────────────────────────────────────────────────
Achieved Occupancy             31.51%       31.56%       15.98%       16.01%
Theoretical Occupancy          33.33%       33.33%       16.67%       16.67%
Waves Per SM                    21.33        21.33        42.67        42.67
──────────────────────────────────────────────────────────────────────────────
Block Limit - Registers           2            2            1            1
Block Limit - Shared Mem          2            2            1            1
Block Limit - SM                 24           24           24           24
Block Limit - Warps               6            6            6            6
Dynamic SMEM Per Block (KB)    40.32        40.32        40.32        40.32
──────────────────────────────────────────────────────────────────────────────
Compute (SM) Throughput        60.98%       60.89%       28.35%       28.29%
Memory Throughput              60.98%       60.89%       25.05%       25.02%
DRAM Throughput                34.57%       34.41%       25.05%       25.02%
L1/TEX Hit Rate                 8.86%        8.84%       12.06%       12.05%
L2 Hit Rate                    13.75%       13.76%       13.34%       13.26%
Warp Cycles Per Issued Instr    8.50         8.50         6.64         6.64
──────────────────────────────────────────────────────────────────────────────
Shared Memory Spilling            0            0            0            0
Avg. Divergent Branches           0            0            0            0
```

## 分析

### 1. v4: COALESCE (C1 vs C4) 对微架构无影响

v4 H6T8C1 和 v4 H6T8C4 的**所有微架构指标完全一致**——occupancy、registers、smem、wave count、
memory/compute throughput、L1/L2 hit rate 无任何差异。Duration 也相同 (395.1 vs 395.7us)。

C4 将 grid 从 1024 缩减到 256 但不改变单 block 的执行特性。nsys 测出的 43% 性能差 (274→393us)
在 ncu 的单次 launch 中未体现——说明 COALESCE 的损失来自 wave scheduling/latency hiding
（多 wave 场景），而非单 block 执行效率。

这也解释了为什么 batch=32 (2048 grid blocks, 85 waves) 时 COALESCE 有一点点收益——wave 数
从 85 降到 11，wave scheduling 开销下降> 单 block 的 pointer 切换开销。

### 2. v5 vs v4: 寄存器压力导致 occupancy 减半——这是核心瓶颈

| 根因指标 | v4 | v5 |
|---------|-----|-----|
| Block Limit Registers | **2** | **1** |
| Achieved Occupancy | **31.5%** | **16.0%** |
| Waves Per SM | 21.33 | **42.67** (2×) |

v5 的 float4 寄存器 (`float4 reg_acc[HEADS]` + 点积展开 + copy_4d 转换) 将每线程寄存器数
从 ~135 推到 >192（65536 / 256 / 1 ≈ 256 regs/thread 上限），迫使 Block Limit Registers 从 2→1。
occupancy 直接减半，wave 数翻倍——每个 SM 只能驻留 1 个 block 而非 2 个。

**v5 的向量化收益 (Warp Cycles Per Issued: 6.64 vs v4 8.50, -22%) 无法弥补 occupancy 减半
带来的 latency hiding 损失**。SM Throughput 从 61% 降到 28%，说明 LSU 和计算单元大量空闲。

### 3. v5 H6T8C1 vs v5 H3T16C4: 寄存器压力相似，均受 Block Limit 限制

两者 Block Limit Registers 均为 1，说明即使 H=3 (reg_acc 从 24→12 float)，寄存器总数仍超
1-block SM 阈值。这可能来自 float4 操作引入的临时寄存器 + copy_4d 的转换指令展开。

v5 H3T16C4 的 Occupancy (16.01%) 与 v5 H6T8C1 (15.98%) 几乎相同，但 nsys 显示前者比后者
在 16×512 配置下快 25%。差异来自 TILE_K=16 减少 sync 频率和 H=3 减少 Q load + 点积次数，
而非 occupancy 差异。

### 4. L1/L2 Cache

v4 的 L1 Hit Rate 9% + L2 Hit Rate 14% ≈ 23% 片上命中。DRAM Throughput 35%，Memory Throughput 61%。
解码注意力是 memory-bound（K/V 从 DRAM 流式读取），v4 的 61% memory throughput 接近极限。

v5 的 Memory Throughput 仅 25%——occupancy 减半导致 memory latency 无法隐藏，访问流水线
空闲。这是 occupancy 瓶颈的直接后果，而非算法问题。

## 改进方向

| 优先级 | 方向 | 预期收益 | 依据 |
|--------|------|---------|------|
| **P0** | **降低寄存器压力** | 恢复 2 block/SM (+100% occupancy) | Block Limit Registers: 1→2 是根因 |
| P0a | `reg_acc` 放回 smem (HEADS≥3 时) | 省 24 float/thread (H=6) | v3 用 smem 存 acc 时 occupancy=62% |
| P0b | 减少 float4 临时寄存器 | 省 10-15 regs | copy_4d 展开 + 点积累加 |
| P1 | 异步 cp.async 与计算重叠加深 | 5-10% | 当前 3-stage, FlashInfer 用更深流水 |
| P2 | COALESCE 仅在大 grid (≥512 blocks) 启用 | 0-5% | 仅在 wave 数足够大时有效 |

核心建议：**v5 的向量化方向正确（Warp CPI -22%），但必须在保持 2 block/SM 的前提下。
短期优先做 P0（reg_acc→smem），目标 25-30% occupancy，而非当前的 16%。**

## 测试脚本

- `test/profile/_ncu_v4v5.py` — ncu profiling target
- `test/profile/_ncu_analyze.sh` — ncu metric extraction
