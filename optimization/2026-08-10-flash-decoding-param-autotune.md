# flash-decoding 参数自动调优

日期: 2026-08-10
测试对象: `src/ops/paged_attention/nvidia/flash_decoding_v2_nvidia.cu`（参数化实验版）
对比基线: 现有 flash_decoding（v1）+ FlashInfer
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 两个正交优化点

### 优化A: grid.y = nh / HEADS（GQA head 共享）

用户建议 grid.y 不限于 nh 或 nkvh，**只要 HEADS 整除 nh，grid.y 就能设为 nh/HEADS**。每个 block 服务 HEADS 个连续 query head，K/V 只 load 一次。

HEADS 个连续 head 落在同一 kv_head 组内才能共享（`kv_head = head / rep`）。对 nh=12, rep=6，有效 HEADS ∈ {1, 2, 3, 6}：

| HEADS | grid.y | 每 block head 数 | K/V load 次数（nh=12） | 含义 |
|-------|--------|-----------------|----------------------|------|
| 1 | 12 | 1 | 12 | v1 基线 |
| 2 | 6 | 2 | 6 | 成对共享 |
| 3 | 4 | 3 | 4 | 3-head 共享 |
| 6 | 2 | 6 | 2 | 完全 GQA 共享（= FlashInfer grid.y=nkvh） |

HEADS=6 时 grid.y=2 即 FlashInfer 架构。每个 consumer warp 持有 HEADS 份 q，对每个 token 算 HEADS 个点积。

### 优化B: 双缓冲粒度 TILE_K（每次 sync 的 token 数）

当前每次 load 4 行 KV，每行只被一个 warp 使用。扩大粒度：每 consumer warp 处理 TILE_K/4 个 token。`__syncthreads` 频率 = 64/TILE_K 次/块。

TILE_K ∈ {4, 8, 16, 32}（TILE_K=32 需 64KB smem，用动态 smem + cudaFuncSetAttribute 支持）。

## 参数扫描（CPU timer, best-of-30）

### 优化A 单独（TILE_K=4 固定）

```
         1x2048   8x1024   16x2048   32x256
H1(v1)    59.2    165.2     607.3    184.0
H2        75.1    145.6     516.6    159.0
H3        48.2    115.9     383.9    112.7   ← 最优
H6        55.8    123.4     418.7    124.8
```
A 单独：**H3 最优**（K/V load 12→4 次，block 数仍够）。H6 因 grid.y=2 block 太少而稍差。

### 优化B 单独（HEADS=1 固定）

```
16x2048:  T4=562.9  T8=577.1  T16=835.1  T32=1412.1
```
B 单独：**几乎无收益**（T8≈T4，T16/T32 大 smem 压低并发）。HEADS=1 时扩大 tile 只增加 load 量不省钱。

### 组合（HEADS × TILE_K）

```
16x2048:  H1/T4=567.5  H3/T8=334.4  H6/T8=303.2  H6/T16=299.5
8x1024:   H1/T4=154.3  H3/T8=102.5  H6/T8=92.0
```
组合最优：**H6/T8（16x2048 用 H6/T16）**。H6 在组合时反超 H3——完全 GQA 共享（K/V load 12→2 次）在 TILE_K=8 的高效 block 内体现带宽节省。

## 最优 vs 基线（组合）

| Config | v1 (H1/T4) | 最优 | 提升 |
|--------|-----------|------|------|
| 1x2048 | 55.9us | 43.6us | -22% |
| 8x1024 | 154.3us | 94.3us | -39% |
| 16x2048 | 567.5us | 300.6us | **-47%** |
| 32x256 | 171.2us | 103.7us | -40% |

vs FlashInfer（16x2048=113us）：从 5.4× 缩到 **2.7×**。

## 全网格 nsys 扫描 + 启发式自动选参

对 batch×totlen（1..32 × 256..2048）全网格 × 12 参数组合做 nsys 精确测量（每 kernel 纯 GPU 耗时）。

### 规律

- **HEADS 由总 KV token 决定**（`tokens = tot_block_num × token_num`）：
  tokens ≤ 512 → H2；≤ 2048 → H3；否则 H6。
- **TILE_K 由并行度（block 总数）决定**：`blocks = tot_block_num × (nh/HEADS)`。
  blocks < 64 → T16（并行不足时 sync 开销占比高）；否则 T8（T16 的 32KB smem 压低每 SM 并发）。

### 决策函数（编译进 C wrapper，传入 heads=0/tile_k=0 触发）

```cpp
static void pick_params(size_t tot_block_num, size_t token_num, size_t nh,
                        int &heads, int &tile_k) {
    const size_t tokens = tot_block_num * token_num;   // 总 KV token
    if (tokens <= 512)      heads = 2;
    else if (tokens <= 2048) heads = 3;
    else                     heads = 6;
    const size_t blocks = tot_block_num * (nh / static_cast<size_t>(heads));
    tile_k = (blocks < 64) ? 16 : 8;
}
```

### 验证（nsys 纯 GPU 耗时）

24 个 config 全部损失 <6%，绝大多数 <1%，**平均损失 0.6%**。

## 修复的 bug

1. **q 加载条件 `is_comp && warp_id < HEADS`**：HEADS=6 > 4 个 consumer warp 时，warp 4-5（producer）被跳过，q[4]/q[5] 未加载 → 垃圾数据。改为 `warp_id < HEADS`（producer 也能帮加载 q）。
2. **pick_params 的 tokens 多乘 batch**：`batch × tot_block × token_num` 里 tot_block_num 已是全批次总块数，重复乘 batch 导致中等 config 选错 HEADS。改为 `tot_block_num × token_num`。
3. **测量方法**：cudaEvent 批量计时对 batch=1 小 kernel 噪声大（曾显示 1×1024 损失 42%），nsys 精确测量后同 config 仅 -0.2%。**结论：小 kernel 性能对比必须用 nsys**。

## 验证

- 全部 16 个 (HEADS, TILE_K) 组合正确性 PASS（max_diff < 0.005）
- 自动选参（heads=0/tile_k=0）24 config 平均损失 0.6%
