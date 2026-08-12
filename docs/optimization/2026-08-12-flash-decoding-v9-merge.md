# flash_decoding v9: v6 架构 + v8 线程布局融合

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v9_nvidia.cu`
基线: v6 (H6T16C1S2U1) / FlashInfer (patched)
硬件: RTX 4060 Laptop

## 设计

融合 v6 与 v8:
- **v6**: grid=(tot_task, nh/HEADS), COALESCE, block 指针预计算, cp.async pipeline,
  reduce kernel (grid 够大)
- **v8**: blockDim=(16, HEADS), 每 thread 8 元素 q/acc, shuffle width=16 归约
  (低寄存器, 高 occupancy)

目标: v6 的 grid 并行度 + v8 的 occupancy, 克服 v6 的 159 regs → 25% 瓶颈。

## ncu (16×512, H6T8C1)

| 指标 | v6 | v9 | FlashInfer |
|------|-----|-----|-----------|
| Registers/Thread | 159 | **114** | 62 |
| Occupancy | 25% | **31.25%** | 62.5% |
| SM Throughput | 49% | 45.95% | 61.6% |
| Short Scoreboard | — | 1.30 | 0.73 |

v9 寄存器 114 (比 v6 降 45), occupancy 31% (提升)。但 SM 吞吐 45.95% (略低于 v6 49%)。

## 性能 (nsys, par+reduce)

```
b×totlen    v6(us)   v9(us)   FI(us)   v9/v6   FI/v9
──────────────────────────────────────────────────────
1×  256     11.5    19.0    10.0   1.65x   0.52x
1× 2048     18.3    23.6    12.3   1.29x   0.52x
1× 4096     32.4    42.7    15.9   1.32x   0.37x
4× 2048     56.1    63.4    25.5   1.13x   0.40x
16× 512     61.4    67.2    25.7   1.09x   0.38x
32× 256     67.0    72.2    25.5   1.08x   0.35x
```

**v9 全面慢于 v6 (8-65%)**。

## 失败原因

v8 的 (16, HEADS) 线程布局移植到 block-per-KV-block 架构, 与 v7 相同问题:

1. **寄存器没降到 v8 的 72**: block-per-KV-block 的固定开销 (task 映射循环,
   block 指针数组, COALESCE 逻辑, cut_idx) 使 reg = 114 而非 72。
   → occupancy 只 31%, shuffle latency 无法被隐藏。

2. **shuffle 归约 overhead**: 每 token 4 次 width=16 shuffle (v6 用 warp 内
   自然归约, 无显式 shuffle)。FlashInfer 能承受是因为 62.5% occupancy 隐藏
   shuffle latency; v9 31% occupancy 暴露。

3. **smem 冗余读 6x**: HEADS 行读同一 k_row (Short Scoreboard 1.30)。

## 结论

**v6 的 warp 内归约 (无 shuffle) 是更高效的 compute**。它的瓶颈是寄存器
(159 → 25% occupancy), 但 compute 本身优于 (16,HEADS) 布局。

**v8/v9 的 (16,HEADS) 布局** 在 block-per-KV-block 里:
- shuffle + 冗余读代价 > occupancy 收益
- 寄存器因固定开销降不到 FlashInfer 的 62

**FlashInfer 的 62 regs 来自 block-per-seq 架构** (每 block 直接 seq, 无 task 映射/
block 指针/COALESCE 固定开销)。v8 单独成功 (72 regs) 正是因为无这些开销,
但 block-per-seq 又受 grid 大小限制。

## 三层架构矛盾总结

| 架构 | 优势 | 瓶颈 |
|------|------|------|
| block-per-KV-block (v6) | grid 大 | reg 159 → occupancy 25% |
| block-per-seq (v8) | reg 72, occ 56% | grid 小 (batch×nkvh) |
| v6+v8 融合 (v9) | 折中 | shuffle + 冗余读 + reg 114 |

**v6 仍是最优自研** (大 batch 快 v3 36%, 小 batch 快 43%)。
逼近 FlashInfer 需完整 partition-kv + 跨 block 协作, 或接受其 2.5x 架构优势。

## 测试脚本

- `test/profile/_test_v9.py` — v9 vs v3 正确性 (25 项 OK)
- `test/profile/_bench_v9.py` — v9 vs v6 vs FlashInfer
- `test/profile/_ncu_v9.py` — v9 occupancy/延迟分析
