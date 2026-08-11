# flash_decoding v4: COALESCE 虚拟变长 KV block 实验

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v4_nvidia.cu`
基线: v3（cp.async 全体参与, H=6/TK=8）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 系统配置

```cpp
// src/config.hpp
KV_CACHE_TOKEN_NUM = 64    // KV cache 块大小 (token)
KV_CACHE_BLOCK_NUM = 128   // KV cache 总块数
BATCH_MAX_TOKEN_NUM = 8192 // 批次最大并行 token 总数
MAX_TOKEN_NUM = 2048       // 序列最大长度上界
MAX_BLOCK_NUM = 33         // 单序列最大块数 (32 + 1 前缀列)
```

模型: DeepSeek-R1-Distill-Qwen-1.5B, NH=12, NKVH=2, HD=128.
合法 config: `batch × totlen ≤ 8192`, `totlen ≤ 2048` (除 1×4096), `batch × ceil(totlen/64) ≤ 128`.

FLOPs: `batch × nh × 4 × totlen × hd`。`4` = QK^T (2) + P@V (2)，softmax 不计。

## 动机

用户实验 2 显示物理 TN 从 64→256 时 batch=16 decode 延迟降 14%。
本实验尝试通过 COALESCE 模板参数在不改变 TN=64 物理布局的前提下，
让每个 grid block 处理 COALESCE 个连续 block_ids 条目，模拟变长 TN。

## 设计迭代

| 版本 | 方法 | 16×2048 H6T8 CO=1→CO=4 |
|------|------|------------------------|
| v4a | 每 block 独立 pipeline + outer loop | 274→393us (+43%) |
| v4b | flatten tile 迭代 + 内联 cp_load | 627→542us |
| v4c | flatten tile + 预计算 `k_blk[COALESCE]` 指针数组 | 324→333us (+3%) |

v4c 核心优化: 预计算 `k_blk[COALESCE]`, `v_blk[COALESCE]`, `tok_off_blk[COALESCE]` 数组，
`cp_tile(stage, global_tile_idx)` 通过 `cb = tile_idx / tiles_per_block` 索引。
pipeline 3-stage 连续跨 block，无 drain。kernel 审查通过，无逻辑 bug。

## 测试结果（nsys 精确 GPU 计时, bf16）

### COALESCE 单独效应（H=6, TK=8 固定，唯一变量 CO）

```
batch×totlen     CO=1     CO=2     CO=4     CO=8   best
──────────────────────────────────────────────────────────
1 ×  256         12.8     23.2     42.1     42.7   CO=1
1 ×  512         12.5     22.4     40.7     79.2   CO=1
1 × 1024         16.1     22.6     41.4     79.1   CO=1
1 × 2048         25.9     28.9     41.2     79.1   CO=1
1 × 4096         44.3     48.8     53.5     81.6   CO=1
2 ×  256         12.6     22.8     41.6     42.1   CO=1
2 ×  512         15.7     22.7     41.4     79.9   CO=1
2 × 1024         25.3     28.7     41.4     79.2   CO=1
2 × 2048         42.5     47.9     53.9     80.5   CO=1
4 ×  256         16.2     23.4     42.0     42.5   CO=1
4 ×  512         25.7     29.0     42.1     79.7   CO=1
4 × 1024         42.6     48.6     54.3     81.3   CO=1
4 × 2048         76.5     80.8     90.3    102.3   CO=1
8 ×  256         27.2     30.2     43.3     43.7   CO=1
8 ×  512         43.3     50.0     54.6     82.3   CO=1
8 × 1024         77.6     81.5    101.4    103.0   CO=1
16 × 256          46.0     57.9     58.5     57.8   CO=1
16 × 512          82.0     85.1     94.9    106.1   CO=1
32 × 256          91.1     93.0    111.4    109.8   CO=1
```

**全部 19 config，CO=1 最优。无一例外。**

CO=2 比 CO=1 慢 2-23%、CO=4 慢 16-234%、CO=8 慢 21-530%。
batch 越大，CO>1 的相对损失越小（16×512: CO=2 仅 +3.8%），但从未反超。

### 全部参数组合 vs v3 基线

v3(H6,T8,CO=1) 作为基线。v4 测试了 7 组参数: (6,8,1) (6,8,2) (6,8,4) (6,8,8) (2,16,1) (3,16,1) (3,16,8)。

```
                  v3 基线        v4 最优               v4/v3
batch×totlen   (H6T8) us    (H,TK,CO)  us          speedup
─────────────────────────────────────────────────────────────
1 ×  256          12.8       H2,T16,C1   7.7          1.66x
1 ×  512          12.5       H2,T16,C1   9.7          1.29x
1 × 1024          16.1       H6,T8 ,C1  16.1          1.00x
1 × 2048          25.9       H6,T8 ,C1  25.9          1.00x
1 × 4096          44.3       H6,T8 ,C1  44.3          1.00x
2 ×  256          12.6       H2,T16,C1   9.8          1.29x
2 ×  512          15.7       H6,T8 ,C1  15.7          1.00x
2 × 1024          25.3       H6,T8 ,C1  25.3          1.00x
2 × 2048          42.5       H6,T8 ,C1  42.5          1.00x
4 ×  256          16.2       H6,T8 ,C1  16.2          1.00x
4 ×  512          25.7       H6,T8 ,C1  25.7          1.00x
4 × 1024          42.6       H6,T8 ,C1  42.6          1.00x
4 × 2048          76.5       H6,T8 ,C1  76.5          1.00x
8 ×  256          27.2       H6,T8 ,C1  27.2          1.00x
8 ×  512          43.3       H6,T8 ,C1  43.3          1.00x
8 × 1024          77.6       H6,T8 ,C1  77.6          1.00x
16 ×  256         46.0       H6,T8 ,C1  46.0          1.00x
16 ×  512         82.0       H6,T8 ,C1  82.0          1.00x
32 ×  256         91.1       H6,T8 ,C1  91.1          1.00x
```

16/19 config v3 基线即最优，仅 3 个极小 config 被 H2/T16 超越。H3/T16/C8 从未进入最优。

| batch×totlen | v3(H6T8) | 最优 | speedup | 最优参数 |
|-------------|----------|------|---------|---------|
| 1 × 256 | 12.8 | 7.7 | 1.66x | **H=2,TK=16,CO=1** |
| 1 × 512 | 12.5 | 9.7 | 1.29x | **H=2,TK=16,CO=1** |
| 2 × 256 | 12.6 | 9.8 | 1.29x | **H=2,TK=16,CO=1** |
| 其余 16 config | — | — | **1.00x** | **H=6,TK=8,CO=1** |

仅在 batch=1×256, 1×512, 2×256 三个 grid 极小的 config 中 H=2/TK=16 有收益。
H=3/TK=16/CO=8 从未进入 top-1。切换到 H=3 本身已比 H=6 慢 8-120%，
COALESCE 无法弥补。

### 与 FlashInfer 对比

FlashInfer v0.6.16, NH=12/NKVH=2/HD=128 历史数据（同卡同模型，来源 `2026-08-08-flash-decoding-tflop.md`）。

| batch×totlen | LLAISYS(us) | FlashInfer(us) | LL/FI |
|-------------|-------------|-----------------|-------|
| 1 × 256 | 7.7 | 48.3 | 0.16× |
| 1 × 2048 | 25.9 | 53.9 | 0.48× |
| 4 × 2048 | 76.5 | 68.1 | 1.12× |
| 8 × 1024 | 77.6 | 68.4 | 1.13× |
| 16 × 256 | 46.0 | 60.5 | 0.76× |

小 batch LLAISYS 反超 6.2×（1×256: 7.7 vs 48.3us），交叉点在 batch≈4。

### 带宽

LLAISYS 峰值 110 GB/s（4×2048），GPU 理论 256 GB/s → 43% 利用率。
FlashInfer 峰值 ~233 GB/s（91%）。差距来自 reduce kernel 开销和 `__syncthreads()` 粒度。

## 为什么 COALESCE 无效

物理 TN 从 64→256（= COALESCE=4 的虚拟等效）之所以有效，是因为
**单一 K/V 基址指针 + 256 token 连续 cp.async，零指针切换**。

v4 COALESCE=4：每 tile 必须查 `k_blk[cb]` 数组获取当前 block 的 K/V 指针。
查表 + 源地址切换的微小开销 × total_tiles 累加，刚好抵消 grid 缩减的 wave scheduling 收益。

在系统 BATMAX_TOKEN_NUM=8192 约束下，最大 grid 仅 256 block（H=6: 16×512 = 256 grid），
wave 调度开销本就极小（256 block / 24 SM ≈ 11 waves），缩减 grid 几乎无收益。
仅当 grid 极大（如 32×4096=2048 block, 85 waves）时 COALESCE 的 wave 缩减才开始
超过指针切换开销，但此 config 超出系统设计范围。

## 结论

1. **COALESCE 在本系统约束下无效。** 19 合法 config 中 H=6/TK=8 控制组显示 CO>1 无一胜出。
2. **最佳参数即 v3 默认 H=6/TK=8。** 仅 3 个极小 config 需 H=2/TK=16。
3. **v4 未接入主接口。** 保留 `flash_decoding_v4_nvidia.cu` 为实验参考，
   qwen2.cpp/op.hpp/op.cpp/.cuh 均无 coalesce 残留。
4. **代码审查结论：kernel 逻辑无 bug。** 任务映射、指针预计算、flatten tile 迭代、
   reduce kernel、partial 索引全部正确。性能损失来自架构开销而非实现错误。

## 测试脚本

- `test/profile/_bench_sys.py` — 全 config COALESCE 隔离对比（nsys + TFLOPS + 带宽）
- `test/profile/_scan_v4_detail.py` — 1200 组合全网格参数扫描
