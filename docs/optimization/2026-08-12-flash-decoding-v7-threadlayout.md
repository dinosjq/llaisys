# flash_decoding v7: FlashInfer 式线程布局尝试

日期: 2026-08-12
测试文件: `src/ops/paged_attention/nvidia/flash_decoding_v7_nvidia.cu`
基线: v6 (H6T16C1S2U1) / FlashInfer (patched group_size=6)
硬件: RTX 4060 Laptop

## 动机

v6 vs FlashInfer ncu 分析显示: FlashInfer 每线程 62 regs (v6 159), occupancy 62.5%
(v6 25%), 10 blocks/SM (v6 3)。核心差异是 FlashInfer 用 blockDim=(16, 6) = 96 threads,
16 threads 沿 head_dim (每 thread 8 元素), 6 = group_size。

v7 尝试在 v6 基础上改用该线程布局:
- blockDim = (16, HEADS), H6 → 96 threads (3 warps)
- 每 thread 8 元素 q/acc 常驻寄存器
- 点积 8 FMA + shuffle width=16 归约
- 去掉 s_q/s_acc (每 head 16 threads 各自归约)

## ncu 对比 (16×512)

| 指标 | v6 | v7 | FlashInfer |
|------|-----|-----|-----------|
| Registers/Thread | 159 | **114** | 62 |
| Occupancy | 25% | **31.25%** | 62.5% |
| SM Throughput | 49% | 45.8% | 61.6% |
| Long Scoreboard | 1.44 | 1.29 | 0.21 |
| Short Scoreboard | — | **1.30** | 0.73 |
| Wait | 1.07 | 1.56 | 0.64 |

寄存器 159→114 (降 45), occupancy 25→31%。但 Short Scoreboard 升高 (1.30)。

## 性能 (nsys, par+reduce)

```
b×totlen    v6(us)   v7(us)   FI(us)   v7/v6   FI/v7
──────────────────────────────────────────────────────
1×  256     11.5    19.0    10.0   1.66x   0.52x
1× 2048     18.2    23.7    12.4   1.30x   0.52x
1× 4096     32.5    43.6    15.9   1.34x   0.36x
4× 2048     56.1    62.3    25.5   1.11x   0.41x
8× 1024     57.0    63.8    25.5   1.12x   0.40x
16× 512     61.7    66.5    25.5   1.08x   0.38x
32× 256     67.8    71.4    25.5   1.05x   0.36x
```

**v7 全面慢于 v6 (5-66%)**, 未接近 FlashInfer。

## 失败原因

把 FlashInfer 的 96-thread 布局移植到 **block-per-KV-block** 架构不兼容:

1. **每 token 串行 + shuffle 归约 overhead**: TILE_K 循环内每 token 做
   `8 FMA + 4 次 width=16 shuffle`。v6 每 warp 并行处理 TILE_K/4 token,
   点积在 warp 内 32 lanes 自然归约 (无显式 shuffle)。v7 每 token 4 shuffle
   是纯 overhead。

2. **smem 冗余读 6x**: 6 行 (head) 读同一 k_row → Short Scoreboard 1.30。
   v6 每 token 的 k_row 只被 1 个 warp 读一次, 供 6 个 head 复用 (寄存器循环)。

3. **每 block 工作量小**: v7 block-per-KV-block 每 block 只处理 64 token,
   compute 短, 固定 overhead (task 映射, pipeline) 摊薄不到。FlashInfer
   block-per-seq 每 block 处理整个 seq 的 KV (数百 token), overhead 均摊。

## 关键结论

**FlashInfer 快的核心不是线程数/寄存器, 而是 block-per-seq 单 kernel 架构**:

1. 每 block 串行遍历整个 seq 的 KV (大量 token), KV 读一次供 group_size 个 head 复用
2. 单 kernel 无 reduce, 无跨 kernel 数据交换
3. 62.5% occupancy 隐藏访存延迟 (带宽打满)

v7 在 block-per-KV-block 里模仿线程布局, 得到:
- occupancy 提升 (25→31%) 但不足以抵消 compute overhead
- 需要 v8 (block-per-seq 单 kernel) 才能真正复现 FlashInfer 优势

## 后续方向

v8: block-per-seq×kv_head 单 kernel。grid=(batch, nkvh), 每 block 服务一个
seq×kv_head, 串行遍历全部 KV token, online softmax 直接出结果, 无 reduce。
这是 FlashInfer 架构, 预期能复现其 62.5% occupancy + 带宽打满。

## 测试脚本

- `test/profile/_test_v7.py` — v7 vs v3 正确性 (25 项 OK)
- `test/profile/_bench_v7.py` — v7 vs v6 vs FlashInfer
- `test/profile/_ncu_v7.py` — v7 occupancy 分析
