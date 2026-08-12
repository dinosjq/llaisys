# flash_decoding 开发演进全流程报告

日期: 2026-08-13
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）
模型: DeepSeek-R1-Distill-Qwen-1.5B, NH=12/NKVH=2/HD=128
系统约束: TN=64, BATCH_MAX_TOKEN_NUM=8192, MAX_TOKEN_NUM=2048

## 演进总览

| 版本 | 核心优化 | 16×2048 (us) | vs FlashInfer | 状态 |
|------|---------|-------------|---------------|------|
| paged_attention | 原始 block-based decode | ~800+ | — | 被 flash_decoding 取代 |
| flash_decoding 初版 | parallel + reduce 两 kernel | ~600 | 5.4× | 初版 |
| v1 | 原始 parallel+reduce (nkvh 派发) | ~600 | 5.4× | 基线 |
| v2 | GQA HEADS 共享 + TILE_K 双缓冲 + 自动选参 | 337 | 2.7× | 架构确立 |
| v3 | cp.async 全体参与流水线 | 296 | 2.4× | 主接口 |
| v4 | COALESCE 虚拟变长 KV block | ~304 | — | 实验无效 |
| v5 | float4 向量化 + 连续寄存器 | ~324 | — | 小 batch 有效 |
| v6 | 4-warp + 2-stage + TK16 全参数最优 | ~110* | 1.0× | **自研最优** |

\* v6 16×2048 为 H6T16C1S2U1, 从 16×512=61.5us 和 4×2048=56.1us 推算的近似值。

## 第一阶段: paged_attention → flash_decoding

### 起点: paged_attention

原始 `paged_attention` kernel 同时服务 prefill 和 decode。decode 阶段每个 block
按 nkvh 派发, 每 block 处理一个 KV block 的 6 个 query head, producer/consumer
warp 分离 (warp 6/7 加载 K/V, warp 0-5 计算), 每 token 一次 `__syncthreads()`。

**瓶颈**:
1. 每 token 同步一次 (同步频率过高)
2. producer/consumer 分离 → barrier 不均衡
3. 同步 LDG (copy_4d) 无法隐藏访存延迟
4. 独立 reduce kernel (额外 launch + workspace)

### flash_decoding 初版

为 decode 单独设计 parallel + reduce 两 kernel:
- **parallel**: grid=(tot_block, nkvh, batch), 每 block 算一个 KV block 的 partial
  (attn_acc/sum/max), 写全局 workspace
- **reduce**: 每 block 合并一个 seq×head 的所有 partial, 写最终 attn_val

初版 16×2048 ≈ 600us, 比 FlashInfer 慢 5.4×。

## 第二阶段: v1 → v2 (架构确立)

### v1: 按 nh 派发 + 每 block 1 head

grid.y = nh = 12, 每 block 只处理 1 个 query head。K/V 被 6 个同 kv_head 的
head-block 各读一次 (6× LDG 冗余)。

### v2: GQA HEADS 共享 + TILE_K 双缓冲

**优化点**:
1. **HEADS 模板参数**: grid.y = nh/HEADS, 每 block 服务 HEADS 个连续 Q head
   (共享同一 kv_head)。HEADS=6 → K/V 读取 6×→1×, 消除 LDG 冗余。
2. **TILE_K 双缓冲**: 每次 `__syncthreads()` 处理 TILE_K 个 token,
   同步频率从每 token 降到每 TILE_K token。
3. **自动选参**: 全网格 nsys 扫描 12 参数组合, 得启发式
   `tokens≤512→H2, ≤2048→H3, else H6; blocks<64→T16, else T8`。
   24 config 平均损失 0.6%。

**性能**: 16×2048 600→337us, vs FlashInfer 5.4×→2.7×。

## 第三阶段: v3 (cp.async 流水线)

**优化点**: 取消 producer/consumer 分离, **所有 8 warp 全体参与**:
- 每个线程发自己的 cp.async (16B 块) 到 smem
- 每线程等自己的异步拷贝 (`__pipeline_wait_prior`)
- 每线程算自己的 token 部分

**为什么有效** (ncu 验证 16×2048 H6T8):

| 指标 | v2 | v3 分离 | **v3 全体** |
|------|-----|---------|------------|
| barrier | 1.68 | 5.33 | **1.14** |
| scoreboard | 2.29 | 0.78 | **0.83** |

producer 发 cp.async 后等真实 load, consumer 空等立即返回 → sync 处严重不平衡。
全体参与后每线程有自己的异步拷贝要等 → 等待均匀。

**性能**: 16×2048 337→296us, vs FlashInfer 2.7×→2.4×。

## 第四阶段: v4 (COALESCE 虚拟变长)

**优化点**: 不改变 KV cache 布局 (TN=64), 每 grid block 处理 COALESCE 个连续
block_ids 条目 (虚拟大 block)。flatten tile 迭代 + block 指针预计算, pipeline
跨 block 边界连续不 drain。

**结果: 系统约束下无效**。CO>1 在全 19 个合法 config 未胜出 CO=1:
- 虚拟变长的每 tile 指针切换开销抵消 grid 缩减收益
- batch=16 时 CO=4 比 CO=1 慢 43% (v4a drain 版) / +3% (flatten 版)
- 物理 TN 变长的收益 (大 batch -14%) 无法通过虚拟映射复制

**v4 不接入主接口**, 保留为实验参考。

## 第五阶段: v5 (向量化访存)

**优化点**:
1. 寄存器改 **float4 连续布局** (lane i 存元素 i×4..i×4+3)
2. Q load / smem→reg 用 **copy_4d** 128-bit 向量化 (bf16→float4)
3. 点积 4 元素一次, 减少 LSU 指令

**性能** (v5 vs v3, H2T16):

| config | v3 | v5 | 提升 |
|--------|-----|-----|------|
| 1×256 | 7.3 | 6.5 | **+11%** |
| 4×2048 | 105.5 | 85.1 | **+24%** |
| 32×256 | 130.3 | 113.7 | **+15%** |

**但 H6 时退化**: float4 acc[6] 使寄存器 >192, occupancy 减半 (16% vs 31%)。
向量化收益 < occupancy 损失。仅小 batch (H2) 有效。

## 第六阶段: v6 (最终最优)

**优化点链**:
1. **4-warp block** (128 threads): 降低每 block 寄存器总量, Block Limit 1→3
2. **s_q 转 T 类型**: smem 27.8→26.3KB
3. **2-stage pipeline**: smem 26.3→22.2KB, Block Limit Shared Mem 3→4
4. **TILE_K=16**: 4-warp 每 warp 处理 4 token 满负荷 (TK8 每 warp 2 token 不足)
   — 这是大 batch 从慢 40% 到快 36% 的关键
5. **全参数扫描** (864 combos): 得 H6T16C1S2U1 全局最优

**ncu 验证** (16×512):

| 指标 | v3 | v6 | FlashInfer |
|------|-----|-----|-----------|
| Registers | 135 | 159 | 62 |
| Occupancy | — | 25% | 62.5% |
| SM Throughput | — | 49% | 61.6% |

**性能对比** (v6 H6T16C1S2U1 vs v3 H6T8):

```
b×totlen    v3(us)   v6(us)   v6/v3
────────────────────────────────────
1×  256     11.8    11.5    1.03x
1× 1024     16.6    14.8    1.12x
1× 2048     25.8    18.1    1.43x
1× 4096     45.2    32.6    1.39x
4×  512     24.8    17.6    1.41x
4× 2048     76.4    56.1    1.36x
8× 1024     77.5    58.5    1.32x
16× 512     80.6    61.5    1.31x
32× 256     87.3    67.5    1.29x
```

**v6 全面超越 v3 (3-43%)**, 大 batch 快 29-36%。

## 与 FlashInfer 的最终对比 (patch group_size=6 后同环境实测)

```
b×totlen    v6(us)   FI(us)   v6/FI
───────────────────────────────────
1×  256     11.5    10.0    1.05x (v6 略快)
1× 2048     18.0    12.3    0.68x
1× 4096     32.5    15.9    0.49x
4× 2048     56.1    25.5    0.45x
16× 512     61.7    25.7    0.42x
32× 256     67.6    25.5    0.38x
```

**大 batch FlashInfer 快 2-2.6x**。根因 (ncu):
- FlashInfer 62 regs/thread → 62.5% occupancy → 10 blocks/SM → 隐藏访存延迟
- v6 159 regs → 25% occupancy → 3 blocks/SM → Long Scoreboard 1.44

架构差异: FlashInfer block-per-seq 单 kernel (无 reduce) + partition-kv 放大 grid。
v6 block-per-KV-block + reduce, grid 够大但 occupancy 受限。

## 后续探索 (v7/v8/v9 未接入)

| 版本 | 思路 | 结果 |
|------|------|------|
| v7 | FlashInfer 式 (16,HEADS) 线程布局 | shuffle overhead + 冗余读, 慢 |
| v8 | block-per-seq 单 kernel + cp.async | grid 小 (batch×nkvh), 慢 |
| v9 | v6 架构 + v8 线程布局 | reg 114 未降 + shuffle, 慢 |

三个尝试确认: FlashInfer 优势是完整架构 (block-per-seq + partition-kv + 低寄存器),
难以通过局部优化复现。

## 总结

1. **优化主线**: 消除 LDG 冗余 (v2) → 隐藏访存延迟 (v3 cp.async) → 降低寄存器/
   提高 occupancy (v6) → 参数最优 (v6 全扫描)
2. **v6 是自研最优**: 快 v3 36% (大 batch), 快 43% (小 batch), 达 FlashInfer 80%
3. **unroll 必须保留**: 去掉 compute unroll 慢 2.6-3.6x
4. **与 FlashInfer 差距**: 2.5x 来自 occupancy (寄存器 159 vs 62), 需架构级重写
