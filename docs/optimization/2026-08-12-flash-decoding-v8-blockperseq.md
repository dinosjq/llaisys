# flash_decoding v8: block-per-seq×kv_head 单 kernel (第一阶段)

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v8_nvidia.cu`
基线: v6 (H6T16C1S2U1) / FlashInfer (patched group_size=6)
硬件: RTX 4060 Laptop

## 设计 (FlashInfer 架构)

- grid = (batch, nkvh), 每 block 服务一个 (seq, kv_head)
- blockDim = (16, rep) = 96 threads, rep = nh/nkvh = 6
- 每 thread 8 元素 q/acc, 点积 8 FMA + shuffle width=16 归约
- 串行遍历该 seq 全部 KV token, online softmax 累积
- 单 kernel, 无 reduce
- **本阶段: 直接读全局 KV (无 smem pipeline), 验证正确性 + 基线**

## ncu (16×512)

| 指标 | v8 | FlashInfer |
|------|-----|-----------|
| Registers/Thread | **72** | 62 |
| Theoretical Occupancy | **56.25%** | 62.5% |
| Long Scoreboard | **8.67** | 0.21 |
| SM Throughput | 11.6% | 61.6% |
| DRAM | 100% | 100% |

寄存器 72 (接近 FlashInfer 62), 理论 occupancy 56.25% (接近 62.5%) — **架构方向正确**。
但实际 warp 执行率 8.35%, Long Scoreboard 8.67 — **每 token 直接读全局 KV, 延迟完全暴露**。

## 性能 (nsys)

```
b×totlen    v6(us)    v8(us)   FI(us)   v8/v6   FI/v8
──────────────────────────────────────────────────────
1×  256     11.4    187.7    10.0   16.43x   0.05x
1× 2048     18.0   1482.3    12.3   82.57x   0.01x
4× 2048     55.7   1481.9    25.5   26.58x   0.02x
16× 512     61.4    374.8    25.4    6.11x   0.07x
32× 256     67.6    194.0    25.4    2.87x   0.13x
```

**v8 慢 v6 3-90x**。直接全局读 KV 且无预取 → 每 token 等全局内存。

## 失败原因

1. **无 cp.async 预取**: FlashInfer 用 cp.async 把 KV 预取到 smem, 隐藏全局延迟。
   v8 直接 LDG, 每 token 等内存 → Long Scoreboard 8.67。
2. **grid 小**: batch×nkvh = 16×2 = 32 blocks (16×512), 喂不满 24 SM。
   FlashInfer 用 partition-kv 拆长 seq, grid 更大。
3. **每 block 串行遍历**: 3 warps 串行遍历 seq 全部 token, warp 间无并行。

## 结论与下一步

**架构方向正确** (regs 72, theoretical occ 56.25% 接近 FlashInfer), 但需要:

1. **cp.async pipeline**: 每 block 双缓冲预取 KV 到 smem, 隐藏全局延迟
   (这是 FlashInfer 能 0.21 Long Scoreboard 的关键)
2. **partition-kv**: 长 seq 拆成多个 block, grid 变大喂满 SM

完整 v8 = block-per-seq + cp.async pipeline + partition。工作量大。
当前 v6 仍是最优自研 (大 batch 快 v3 36%, 小 batch 快 43%)。

## 测试脚本

- `test/profile/_test_v8.py` — v8 vs v3 正确性 (5 项 OK)
- `test/profile/_bench_v8.py` — v8 vs v6 vs FlashInfer
- `test/profile/_ncu_v8.py` — v8 occupancy/延迟分析
