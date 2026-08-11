# flash-decoding 接口重构 + 性能对比

日期: 2026-08-10
测试对象: `src/ops/paged_attention/nvidia/flash_decoding_nvidia.cu`（接口重构后）
对比基线: FlashInfer 0.6.16 `BatchDecodeWithPagedKVCacheWrapper`（同卡、同配置、同输入）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s 显存带宽, Ada Lovelace CC 8.9）

## 接口变更（2026-08-10）

`paged_attention` 新增参数 `tot_block_num`（整个批次所有序列使用的总 KV 块数）。

decode 的 flash_decoding 网格从 3D `(max_block_num, nh, batch_size)` 改为 **2D `(tot_block_num, nh)`**：
- `blockIdx.x` 是全局块索引 `[0, tot_block_num)`，覆盖所有序列的所有块
- kernel 内部通过 block_ids 列 0 的**累计前缀和**反推 `(batch_id, local_table_id)`
- `blockIdx.y = nh` 不变

模板化 `_d`/`_dv`（默认 128），launch 按运行时 d/dv 分派实例化 `{8,16,32,64,128}`。

### 修复的 bug

1. **前缀查找下溢**：原循环从 `i=1` 开始，`table_id=0 < prefix[1]` 误捕 batch 0 → 下溢成巨大数 → batch 0 结果全错（batched 测试 sample 0 diff=0.57，其余完美）。修复：从 `i=0` 开始，batch 0 不减前缀。
2. **`_block_ids[i]` 索引错误**：2D 布局下列 0 前缀和在 `[i * max_block_num]`，非 `[i]`。
3. **模板未实例化**：launch 未传 d/dv → 小 hd（8/16/64）按 128 越界读全零。加 (d,dv) 分派。
4. **`reg_acc[_dv>>5]` 零长度数组**：dv<32 时大小为 0，改为 `(_dv+31)>>5`。
5. **qwen2.cpp `tot_block_num` 越界**：`block_ids[batch_size*MAX_BLOCK_NUM]` 越界一行，改为 `[(batch_size-1)*MAX_BLOCK_NUM]`（末行前缀和）。

## 测试配置

模型 DeepSeek-R1-Distill-Qwen-1.5B: nh=12, nkvh=2, hd=128, `TOKEN_NUM=64`。
FLOPs = `batch × nh × 4 × totlen × hd`。bf16。

## 性能结果（CPU timer，best-of-50）

```
batch x totlen  LLAISYS(us)  FlashInfer(us)  LLAISYS/FlashInfer
──────────────────────────────────────────────────────────────
  1 x 256         16.4          40.2            0.41×  (快 2.4×)
  1 x 512         31.7          40.4            0.78×
  1 x 1024        28.7          40.7            0.70×
  1 x 2048        48.1          41.1            1.17×
  4 x 2048       154.6          54.9            2.8×
  8 x 2048       301.2          72.8            4.1×
 16 x 2048       604.2         111.0            5.4×
  mean           139.2            —                —
```

## 与接口重构前对比

| config | 重构前 | 重构后 | 提升 |
|--------|-------|-------|------|
| 1×256 | 33.8us | **16.4us** | -51% |
| 1×2048 | 75.5us | **48.1us** | -36% |
| 16×2048 | 822.5us | **604.2us** | -27% |

## 分析

**小 batch 完胜 FlashInfer**（1×256 快 2.4×，1×2048 打平）。batch=1 时新 2D 网格 `(tot_block_num, nh)` 产生 48-384 个 block，把 24 SM 塞满；FlashInfer 的 `(batch, nkvh)` 只有 2 个 block，靠 block 内 cp.async 流水弥补。

**大 batch 仍有 2.8-5.4× 差距**（16×2048 慢 5.4×）。根因是同步 load（`copy_4d` 同步全局读 → smem）+ 每 token `__syncthreads`，block 内 load/compute 串行。FlashInfer 用 `cp.async` 深流水 + partition-kv 让每个 block 内部跑满。这是下一步优化方向（cp.async pipeline + partition-kv + MergeStates）。

## 验证

- `test/ops/flash_decoding.py --device nvidia` PASS（含单样本 + batched + 多 dtype）
- `test/ops/paged_attention.py --device nvidia` PASS（prefill 路径回归）
- `test/test_runtime.py --device nvidia` PASS
- benchmark 正确性 PASS（per-head maxdiff < 容差）
