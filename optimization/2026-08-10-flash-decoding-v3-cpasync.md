# flash-decoding v3: cp.async 全体参与流水线

日期: 2026-08-10
测试对象: `src/ops/paged_attention/nvidia/flash_decoding_v3_nvidia.cu`
对比基线: v2（同步双缓冲 + producer/consumer 分离）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 版本演进

| 版本 | 核心 | 16×2048 | 说明 |
|------|------|---------|------|
| v1 | 原始并行+reduce | ~600us | 用户初版 |
| v2 | GQA HEADS 共享 + TILE_K 双缓冲 + 自动选参 | 337us | 同步 load，producer/consumer 分离 |
| v3 | **cp.async 全体参与流水线** | **296us** | 本版 |

## v3 架构

基于 v2 结构（grid=(tot_block, nh/HEADS)、HEADS 个 head 共享 kv_head、3-stage 缓冲），
但**取消 producer/consumer 分离**——所有 8 个 warp 双角色：

1. **load**：所有 256 线程按 tid 线性分摊发 cp.async（16B 块）到 smem
2. **wait**：每个线程等自己的异步拷贝（`__pipeline_wait_prior`）
3. **compute**：每 warp 处理 TILE_K/8 个 token × HEADS head
4. **归约**：8-warp 归约到 warp 0

smem 存**原始 T 类型**（cp.async 直拷 16B，不转 float），计算时 cast。

## 为什么取消 producer/consumer

v2/v3 分离版用 4 producer warp（同步 load）+ 4 consumer warp（计算）。
ncu 显示分离版 cp.async（v3 第一版）**barrier 从 1.68 飙到 5.33**：
producer 发 cp.async 后要 wait（等真实 load），consumer 空 wait 立即返回
→ sync 处 producer 慢、consumer 空等 → 严重不平衡。

取消分离后，**每个线程都有自己的异步拷贝要等** → sync 处等待均匀。
ncu 验证（16×2048, H6T8）：

| 指标 | v2 | v3 分离 | **v3 全体** |
|------|-----|---------|------------|
| barrier | 1.68 | 5.33 | **1.14** |
| scoreboard (L2 load) | 2.29 | 0.78 | **0.83** |

两个 stall 同时降低——cp.async 隐藏了 L2 load 延迟，均匀等待消除了 barrier 瓶颈。

## 归约优化实验（ncu 精确对比，16×2048）

三种 8-warp 归约方案：

| 方案 | 关键路径 | 耗时 | barrier |
|------|---------|------|---------|
| 线性（warp0 串行并 1..7） | 7 次 | 342us | 1.14 |
| 树形（stride 1→2→4，3 轮） | 3 轮 | 369us | 1.41 |
| **warp0/1（并行分半）** | 3 并行 + 1 | **337us** | 1.25 |

- **树形无益**：每轮 sync + smem 写回开销超过关键路径减短
- **warp0/1 最优**：warp0 并 2-4，warp1 并 5-7（并行 3 次），最后 warp0 并 warp1
  （1 次）。只 2 次 sync，关键路径 7→4。ncu 比线性省 1.5%。

## 性能对比（v3 自动选参 vs v2）

```
batch totlen    v2(us)   v3(us)
  1    256       29.7     28.3
  1   1024       36.1     38.4   (+6%)
  1   2048       48.2     51.6   (+7%)
  4   2048       97.9     90.9
  8   1024      100.7     93.4
  8   2048      169.8    159.3
 16   2048      327.2    296.5   (-9.4%)
 32   256       110.3     98.5   (-11%)
```

v3 全面优于 v2（除 1×1024/2048 小 batch 略差 6-7%，大 batch 提升 6-11%）。

## 最终主接口性能（v3 接管）

mean: **76.8us**（v2 81.4），16×2048: **304us**（v2 337）。

## 接入状态

- v3 实现 `nvidia::flash_decoding()`（主接口，自动选参），op.cpp 无需改动
- v1/v2 备份在 `src/ops/paged_attention/nvidia/backup/`
  （`flash_decoding_nvidia_v1.cu/.cuh`, `flash_decoding_nvidia_v2.cu`）
- v2 文件保留 kernel + C wrapper（`llaisysFlashDecodingV2`）供实验/对比
- 正确性: flash_decoding / paged_attention / runtime 测试全 PASS
