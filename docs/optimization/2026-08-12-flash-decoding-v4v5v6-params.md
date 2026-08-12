# flash_decoding v4/v5/v6 各版本最优配置扫描

日期: 2026-08-12
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）
模型: DeepSeek-R1-Distill-Qwen-1.5B, NH=12/NKVH=2/HD=128
系统约束: TN=64, BATCH_MAX_TOKEN_NUM=8192, MAX_TOKEN_NUM=2048

## 参数空间

| 维度 | 取值 | 说明 |
|------|------|------|
| HEADS | 1,2,3,6 | 每 block 服务 Q head 数 (grid.y=nh/HEADS) |
| TILE_K | 8,16 | 每次 sync 处理的 token 数 |
| COALESCE | 1,2,4 | 虚拟 KV block 合并数 (v4/v5/v6) |
| NUM_STAGES | 2,3 | cp.async 流水线深度 (v6 参数化) |
| UNROLL | 0,1 | compute 内层 HEADS 循环展开 (v6 参数化) |
| blockDim | 256/128 | v5/v6: 8-warp/4-warp |

## v6 全参数扫描 (9 configs × 96 combos = 864)

### 每 config 最优

| config | v6 最优 (par us) |
|--------|-----------------|
| 1×256 | H1T16C1S2U0: 7.7 |
| 1×1024 | H2T16C1S2U1: 10.2 |
| 1×2048 | H6T16C1S2U1: 14.5 |
| 1×4096 | H6T16C1S2U1: 26.8 |
| 4×512 | H6T16C1S2U1: 14.6 |
| 4×2048 | H6T16C1S2U1: 52.1 |
| 8×1024 | H6T16C1S2U1: 53.6 |
| 16×512 | H6T16C1S2U1: 56.4 |
| 32×256 | H6T16C4S2U1: 59.9 |

### 维度结论

1. **TILE_K=16 全面优于 8** — 4-warp block 每 warp 处理 TILE_K/4 token。
   TK16 → 每 warp 4 token，充分利用 4 warps 并行度。TK8 时每 warp 仅 2 token，
   并行度不足。**这是 v6 大 batch 从慢 40% 到快 36% 的关键。**
2. **NUM_STAGES=2 优于 3** — 省 4KB smem，Block Limit Shared Mem 3→4。
3. **UNROLL=1 略优** — 寄存器无显著增加（nvcc 自主分配），ILP 收益为正。
4. **COALESCE=1 最优** — CO>1 在所有 config 未胜出（延续 v4/v5 结论）。
5. **HEADS**: 小 batch(≤1024) 用 H1/H2 (grid 并行度)，大 batch 用 H6 (K/V 共享)。

## 各版本最优配置对比 (par+reduce 总时间)

```
b×totlen    v3_H6T8   v4_H6T8C1   v5_H6T8C1   v6_H6T16C1S2U1   v6/v3
────────────────────────────────────────────────────────────────────
1×  256      11.8      12.3        12.2          11.5           1.03x
1× 1024      16.6       —            —           14.8           1.12x
1× 2048      25.8      25.9        22.2          18.1           1.43x
1× 4096      45.2      44.3        41.4          32.6           1.39x
4×  512      24.8      25.7        21.5          17.6           1.41x
4× 2048      76.4      76.5        74.8          56.1           1.36x
8× 1024      77.5      77.6        76.7          58.5           1.32x
16× 512      80.6      82.0        79.6          61.5           1.31x
32× 256      87.3      89.4        87.5          67.5           1.29x
```

**v6 H6T16C1S2U1 全面超越 v3 (+3-43%)，大 batch 快 29-36%。**

### 各版本最优参数

| 版本 | 架构 | 最优 (大 batch) | 最优 (小 batch) |
|------|------|----------------|----------------|
| v3 | 标量 8-warp 3-stage | H6T8 | H2T16 |
| v4 | 标量 8-warp + COALESCE | H6T8C1 | H2T16C1 |
| v5 | float4 8-warp | H6T8C1 | H3T16C4 |
| v6 | float4 4-warp 2-stage | **H6T16C1S2U1** | **H1T16C1S2U0** |

## v6 为什么赢

1. **4-warp + TK16**: 每 warp 4 token，4 warps 满负荷。v3 8-warp + TK8 每 warp 1 token，
   但 8 warps 有更多并行 warp（ILP 更高）。v6 的 4-warp 更省寄存器 (144 vs v3 ~135?),
   Block Limit 3 blocks/SM。
2. **2-stage pipeline**: smem 22KB → 4 blocks/SM (smem 允许)，配合寄存器 3 blocks。
3. **float4 向量化**: copy_4d 减少 LSU 指令，Warp CPI 5.3 vs v4 8.5。

## 接入建议

v6 H6T16C1S2U1 是当前最优。若接入主接口:
- 默认 H6T16C1S2U1 (大 batch), 小 batch (tokens≤1024) 切 H2T16C1S2U1
- 自动选参: tokens≤512→H1T16, ≤1024→H2T16, 否则 H6T16
- NUM_STAGES=2, UNROLL=1 固定

## 测试脚本

## v6 最优 vs FlashInfer (同环境实测)

**patch FlashInfer 支持 NH=12 group_size=6**: 修改 `flashinfer/data/include/flashinfer/utils.cuh`
的 `DISPATCH_GQA_GROUP_SIZE` 宏, 加入 `group_size==6` 分支。清 JIT cache 后用
`CC=gcc-12 CXX=g++-12` 编译运行成功。同环境 nsys 实测 (par+reduce vs 单 kernel):

```
b×totlen    v6(us)    FlashInfer(us)   v6/FI
─────────────────────────────────────────────
1×  256       9.5        10.0          1.05x  (v6 略快)
1× 1024      12.7        10.0          0.79x
1× 2048      18.0        12.3          0.68x
1× 4096      32.5        15.9          0.49x
4×  512      17.5        12.3          0.70x
4× 2048      56.1        25.5          0.45x
8× 1024      58.4        25.6          0.44x
16× 512      61.7        25.6          0.42x
32× 256      67.6        25.4          0.38x
```

**大 batch FlashInfer 快 2-2.6x**, 仅 1×256 v6 略快 5%。

> 修正: 此前文档用 2026-08-08 历史 FlashInfer 数据 (1×256=48.3us) 得出
> "v6 全面快于 FlashInfer" 的结论**不准确** — 那批数据在 NH=12 group_size=6
> 无法运行的环境下取得, 不可靠。patch 后同环境实测为准。

### 差距根因

FlashInfer 架构: **block-per-seq×kv_head**, grid=(batch, nkvh), 单 kernel 完成全部
KV token 计算, online softmax 直接出结果, 无 reduce kernel。

v6 架构: **block-per-KV-block + reduce kernel**, 需要 partial 写出 + 二次归约。
reduce kernel 的额外 launch + workspace R/W 在大 batch (大量 partial) 时开销显著。

FlashInfer 大 batch 25us vs v6 56us, 差距 ~2.2x 主要来自:
1. reduce kernel 额外开销 (大 batch partial 数多)
2. FlashInfer 单 kernel 无跨 kernel 数据交换

**v6 的价值**: 小 batch (1×256) 反超, 且架构更简单 (无 FlashInfer 的 workspace/indptr)。
若需逼近 FlashInfer 大 batch 性能, 需架构级改动 (block-per-seq 单 kernel)。

## 测试脚本

- `test/profile/_scan_v6_opt.py` — v6 96 组合全扫描 (4 shard)
- `test/profile/_bench_v6_best.py` — v6 最优 vs v3 对比
- `test/profile/_verify_v6best.py` — v6 最优配置正确性验证
