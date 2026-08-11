# flash_decoding v4: COALESCE 虚拟变长 KV block 实验

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v4_nvidia.cu`
基线: v3（cp.async 全体参与, H=6/TK=8/CO=1）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 系统配置

```cpp
// src/config.hpp
KV_CACHE_TOKEN_NUM = 64    // kv cache 块大小 (token)
KV_CACHE_BLOCK_NUM = 128   // kv cache 总块数
BATCH_MAX_TOKEN_NUM = 8192 // 批次最大并行 token 总数
MAX_TOKEN_NUM = 2048       // 序列最大长度上界
MAX_BLOCK_NUM = 33         // 单序列最大块数 (32 + 1 前缀)
```

模型: DeepSeek-R1-Distill-Qwen-1.5B, NH=12, NKVH=2, HD=128.

FLOPs: `batch × nh × 4 × totlen × hd`。`4` = QK^T (2) + P@V (2)，softmax 不计。

合法 config 约束: `batch × totlen ≤ 8192`, `totlen ≤ 2048`, `batch × ceil(totlen/64) ≤ 128`.

## 动机

用户实验 2 显示将物理 TN 从 64 提升到 128/256 时，大 batch decode 有 9-14% 延迟降低。
本实验尝试通过 COALESCE 模板参数在不改变 KV cache 物理布局（TN=64）的前提下模拟
变长 TN：CPU 预计算 `tot_task_num = sum ceil(block_num_i / COALESCE)`，kernel 内
每个 grid block 把 COALESCE 个连续 block_ids 条目当作一个虚拟大 block 处理。

## 设计迭代

| 版本 | 问题 | 16×2048 H6T8 CO=1→CO=4 |
|------|------|------------------------|
| v4a | 每 block 边界 drain pipeline | 274→393us (+43%) |
| v4b | flatten tile + 内联 cp_load，每 tile 重算 K/V 指针 | 627→542us |
| v4c | 预计算 `k_blk[COALESCE]` 指针数组 + cp_tile lambda | 324→333us (+3%) |

v4c 核心: 预计算 `k_blk[COALESCE]`, `v_blk[COALESCE]`, `tok_off_blk[COALESCE]` 三个数组，
`cp_tile(stage, global_tile_idx)` 通过 `cb = tile_idx / tiles_per_block` 索引数组。
pipeline 3-stage 连续跨 block，不 drain。

## 测试结果（nsys 精确 GPU 计时, bf16）

### 全部合法 config（18 个）

| batch × totlen | v4 (us) | TFLOPS | GB/s | 最优 (H,TK,CO) | grid |
|---------------|---------|--------|------|-----------------|------|
| 1 × 256 | 7.8 | 0.20 | 34.4 | **H=2 TK=16 CO=1** | 24 |
| 1 × 512 | 9.7 | 0.32 | 54.5 | **H=2 TK=16 CO=1** | 48 |
| 1 × 1024 | 16.0 | 0.39 | 65.7 | H=6 TK=8 CO=1 | 32 |
| 1 × 2048 | 26.0 | 0.48 | 81.0 | H=6 TK=8 CO=1 | 64 |
| 2 × 256 | 9.9 | 0.32 | 54.3 | **H=2 TK=16 CO=1** | 48 |
| 2 × 512 | 15.8 | 0.40 | 67.2 | H=6 TK=8 CO=1 | 32 |
| 2 × 1024 | 25.3 | 0.50 | 83.3 | H=6 TK=8 CO=1 | 64 |
| 2 × 2048 | 42.7 | 0.59 | 98.6 | H=6 TK=8 CO=1 | 128 |
| 4 × 256 | 16.1 | 0.39 | 66.8 | H=6 TK=8 CO=1 | 32 |
| 4 × 512 | 25.8 | 0.49 | 82.3 | H=6 TK=8 CO=1 | 64 |
| 4 × 1024 | 42.4 | 0.59 | 99.6 | H=6 TK=8 CO=1 | 128 |
| 4 × 2048 | 76.5 | 0.66 | 110.0 | H=6 TK=8 CO=1 | 256 |
| 8 × 256 | 27.1 | 0.46 | 79.2 | H=6 TK=8 CO=1 | 64 |
| 8 × 512 | 43.0 | 0.59 | 98.7 | H=6 TK=8 CO=1 | 128 |
| 8 × 1024 | 78.2 | 0.64 | 107.9 | H=6 TK=8 CO=1 | 256 |
| 16 × 256 | 46.4 | 0.54 | 92.6 | H=6 TK=8 CO=1 | 128 |
| 16 × 512 | 82.3 | 0.61 | 103.1 | H=6 TK=8 CO=1 | 256 |
| 32 × 256 | 89.6 | 0.56 | 95.8 | H=6 TK=8 CO=1 | 256 |

**H=6/TK=8/CO=1（v3 默认）在 15/18 config 中最优。**
仅 batch=1×256, 1×512, 2×256 三个极小 config 受益于 H=2/TK=16（grid 太小，提高并行度）。

**CO=8（H=3/TK=16/CO=8）在全部 18 config 中未赢一次。**

### vs v3 默认

```
batch×totlen    v3(H6T8)   best      speedup  best_cfg
────────────────────────────────────────────────────────
1 ×  256          12.7       7.8      1.63x    H=2 TK=16 CO=1
1 ×  512          12.4       9.7      1.28x    H=2 TK=16 CO=1
2 ×  256          12.5       9.9      1.26x    H=2 TK=16 CO=1
其余 15 config           均 1.00x            H=6 TK=8 CO=1
```

### 带宽利用率

峰值 110.0 GB/s（4×2048），GPU 理论 256 GB/s → 43%。FlashInfer 峰值 ~233 GB/s（91%，来自
`2026-08-08-flash-decoding-tflop.md` 数据，同卡同模型）。

差距根因: (1) reduce kernel 额外 launch + workspace R/W，(2) 每 8/16 token 一次
`__syncthreads()` vs FlashInfer 的 tile 级同步。

## 为什么 COALESCE 无效

物理 TN=256 和 v4 COALESCE=4 虽都处理 256 token/task，但有一个关键区别:

**物理 TN=256**: 单一 K/V 基址指针，256 token 连续 cp.async，零指针切换。
**v4 COALESCE=4**: 4 个物理 block × 64 token，每 tile 按 `cb = global_tile / tiles_per_block`
索引 `k_blk[cb]`，cp.async 源地址每 block 不同。

查表 + 地址切换的微小开销 × total_tiles（32 tiles for CO=4）累加，抵消了 grid 缩减的
wave scheduling 收益。在系统约束下最大 grid 仅 256 block（16×512 H=6），wave 调度
开销本就很小，缩减 grid 几乎无收益。

## 与 FlashInfer 对比

FlashInfer 数据来自 `2026-08-08-flash-decoding-tflop.md`（同卡、同 NH=12/NKVH=2/HD=128、
同测试方法）。FlashInfer v0.6.16 不支持 NH=12 的 group_size=6，无法在本环境复测，
故引用已验证的历史数据。

| batch×totlen | LLAISYS(us) | FlashInfer(us) | LL/FI |
|-------------|-------------|-----------------|-------|
| 1 × 256 | 7.8 | 48.3 | 0.16× |
| 1 × 512 | 9.7 | 55.0 | 0.18× |
| 1 × 1024 | 16.0 | 48.8 | 0.33× |
| 1 × 2048 | 26.0 | 53.9 | 0.48× |
| 4 × 512 | 25.8 | 53.6 | 0.48× |
| 4 × 1024 | 42.4 | 58.0 | 0.73× |
| 4 × 2048 | 76.5 | 68.1 | 1.12× |
| 8 × 512 | 43.0 | 56.2 | 0.77× |
| 8 × 1024 | 78.2 | 68.4 | 1.14× |
| 16 × 256 | 46.4 | 60.5 | 0.77× |
| 16 × 512 | 82.3 | 72.8 | 1.13× |

**小 batch LLAISYS 反超**: 1×256 快 6.2×（7.8 vs 48.3us）。FlashInfer 有框架层
固定开销（workspace、indptr/indices），batch=1 时远超实际计算。

**交叉点在 batch≈4**: 4×2048 基本持平（76.5 vs 68.1, 1.12×）。更大 batch 后
FlashInfer 单 kernel + cp.async tile 流水优势显现，但差距已在 1.1-1.2× 以内。

架构差异: LLAISYS block-per-KV-block + reduce kernel vs FlashInfer block-per-seq×kv_head
单 kernel。

## 结论

1. **COALESCE 在本系统约束下无效。** 18 合法 config 中 CO=8 从未击败 H=6/TK=8/CO=1。
   虚拟变长的每 tile 指针切换开销抵消了 grid 缩减收益。

2. **当前最优参数就是 v3 默认。** H=6/TK=8/CO=1 覆盖 15/18 config。仅 batch=1×256,
   1×512, 2×256 三个极小 config 需要 H=2/TK=16（grid 并行度不足）。

3. **v4 不接入主接口。** 保留 `flash_decoding_v4_nvidia.cu` 为实验参考。qwen2.cpp
   不包含 coalesce 调度逻辑。接口无 coalesce 参数。

4. **与 FlashInfer 差距已大幅缩小。** v1 时代差 5.4×（16×2048），当前 v3 仅差 1.1-1.2×
   （4×2048, 8×1024），小 batch 甚至反超 6.2×。剩余差距来自 reduce kernel 开销和
   同步粒度，需架构级改动（单 kernel block-per-seq）才能弥合。

## 测试脚本

- `test/profile/_bench_sys.py` — 系统约束内全部合法 config benchmark（nsys + TFLOPS）
- `test/profile/_scan_v4_detail.py` — 全网格 1200 组合参数扫描
- `test/profile/_fi_vs_ll.py` — FlashInfer vs v4 对比基准
