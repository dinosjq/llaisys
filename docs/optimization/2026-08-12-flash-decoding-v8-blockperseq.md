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

## 第二阶段: cp.async pipeline

给 v8 加 2-stage cp.async 双缓冲预取 KV tile 到 smem。

### ncu 变化 (16×512)

| 指标 | v8 直接读 | v8 + cp.async |
|------|----------|--------------|
| Long Scoreboard | 8.67 | **1.01** |
| SM Throughput | 11.6% | 17.2% |
| Warps Active | 8.35% | 8.40% |

**cp.async 成功隐藏全局延迟 (Long Scoreboard 8.67→1.01)**, 但 SM 吞吐仅 17.2%,
warps active 8.4% — **grid 太小是根本瓶颈**。

### 性能 (v8+cp.async vs v6 vs FlashInfer)

```
b×totlen    v6(us)  v8pipe(us)  FI(us)   v8pipe/v6
──────────────────────────────────────────────────
1×  256     11.5     117.2      10.0     10.2x
4× 2048     56.4     921.2      25.5     16.3x
16× 512     61.0     248.9      25.7      4.1x
32× 256     67.9     135.9      25.5      2.0x
```

cp.async 改善 (187→117, 1481→921) 但仍慢 v6 2-50x。

## 最终结论: 三个瓶颈的层级

v8 探索完整回答"为什么 FlashInfer 快 / 如何逼近":

| 层级 | 瓶颈 | 状态 |
|------|------|------|
| 1. 寄存器/occupancy | v6 159regs→25%, v8 72regs→56% | ✅ 已解决 (v8 布局) |
| 2. 访存延迟 | Long Scoreboard 8.67→1.01 | ✅ 已解决 (cp.async) |
| 3. **并行度/grid** | batch×nkvh 太小, SM 吞吐 17% | ❌ 需 partition-kv |

**block-per-seq 的 grid = batch×nkvh 是根本限制**。小 batch 长 seq (如 1×4096)
只有 2 blocks, SM 上活跃 warp 极少, 无法打满带宽。

FlashInfer 用 **partition-kv**: 长 seq 拆成多个 block (每 block 处理部分 KV),
grid = batch×nkvh×partition, 放大并行度。拆块后每 block 算部分结果, 需跨 block
合并 (写 partial + reduce, 或原子/锁) — 这实际上回到类似 v6 的 reduce 架构。

**v6 已是 partition-kv 的一种形式** (block-per-KV-block = 每 64 token 一块,
grid 大, reduce 合并)。v6 的瓶颈是 occupancy (159 regs → 25%), 不是 grid。

## 完整逼近 FlashInfer 的路线

```
最优 = v6 架构 (block-per-KV-block + reduce, grid 够大)
     + v8 的低寄存器布局 (72 regs, 高 occupancy)
     + 每 block 处理多 KV block (摊薄固定开销)
     - 需解决: v7 尝试过的 shuffle overhead + smem 冗余读
```

或完整实现 FlashInfer partition-kv + 跨 block 协作 (大工程)。

当前 v6 仍是最优自研 (大 batch 快 v3 36%, 小 batch 快 43%)。

## 测试脚本

- `test/profile/_test_v8.py` — v8 vs v3 正确性 (5 项 OK)
- `test/profile/_bench_v8.py` — v8 vs v6 vs FlashInfer
- `test/profile/_ncu_v8.py` — v8 occupancy/延迟分析
