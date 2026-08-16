# Flash Decoding 优化汇总

> 通读并合并 `archive/flash-decoding/` 下全部专题报告后重写。  
> 原文按版本保留在同目录 archive，便于核对细节与复跑脚本。  

**硬件：** RTX 4060 Laptop（24 SM，256 GB/s，Ada CC 8.9）  
**模型：** DeepSeek-R1-Distill-Qwen-1.5B（nh=12，nkvh=2，hd=128）  
**系统：** `KV_CACHE_TOKEN_NUM=64`，`BATCH_MAX_TOKEN_NUM=8192`，`MAX_TOKEN_NUM=2048`  
**对照：** FlashInfer 0.6.16 `BatchDecodeWithPagedKVCacheWrapper`。  
**公平口径：** FlashInfer 当时 **未提供 `group_size=6`**（Qwen-1.5B：`nh/nkvh=12/2`）；仅在其 **原生已支持的 GQA 配置** 上对照时，自研 v6 **基本逼近 FlashInfer**。下文「patch 后 gs=6」表是强行补齐该形状后的研究对照，**不作主结论**。

## 目录

1. [演进总览](#1-演进总览)
2. [早期分析与接口重构（→v1/v2）](#2-早期分析与接口重构)
3. [v2：GQA HEADS + TILE_K 自动选参](#3-v2gqa-heads--tile_k-自动选参)
4. [v3：cp.async 全体参与](#4-v3cpasync-全体参与)
5. [v4：COALESCE（废弃）](#5-v4coalesce废弃)
6. [v5：向量化与寄存器压力（废弃）](#6-v5向量化与寄存器压力废弃)
7. [v6：4-warp + 参数最优（自研最优）](#7-v64-warp--参数最优自研最优)
8. [v7/v8/v9：逼近 FlashInfer 的失败路径](#8-v7v8v9逼近-flashinfer-的失败路径)
9. [与 FlashInfer 最终对比与结论](#9-与-flashinfer-最终对比与结论)
10. [原文索引](#10-原文索引)

---

## 1. 演进总览

| 版本 | 核心优化 | 16×2048 量级 | vs FlashInfer | 状态 |
|------|---------|-------------|---------------|------|
| paged_attention | FA2，query 方向并行 | ~800+ us | — | decode 并行度浪费 |
| v1 | KV 方向 parallel+reduce；后改 nkvh 派发 / 2D grid | ~600 us | ~5.4× | 基线 |
| v2 | GQA `HEADS` 共享 + `TILE_K` + 启发式选参 | ~327 us | ~2.7× | 架构确立 |
| v3 | cp.async **全体参与**流水线 | **296 us** | ~2.4× | 曾接主接口 |
| v4 | COALESCE 虚拟变长 KV block | 无效（CO>1 全面退化） | — | **废弃** |
| v5 | float4 向量化 | 小 batch 有收益，大 batch 退化 | — | **废弃** |
| v6 | 4-warp + 2-stage + TK16 + 全参扫描 | 大 batch 快 v3 ~30% | **FI 原生配置下基本逼近**；patch gs=6 后大 batch 仍慢 ~2×（非公平主结论） | **自研最优** |
| v7–v9 | FI 线程布局 / block-per-seq / 融合 | 全面慢于 v6 | — | **废弃** |

统一 nsys 实测（来源 `v4v5v6-params`；各版最优参，par+reduce；v3=H6T8，v4=H6T8C1，v5=H6T8C1，v6=H6T16C1S2U1）：

```
b×totlen    v3_H6T8   v4_H6T8C1   v5_H6T8C1   v6_H6T16C1S2U1   v6/v3
1×  256      11.8      12.3        12.2          11.5           1.03×
1× 1024      16.6       —            —           14.8           1.12×
1× 2048      25.8      25.9        22.2          18.1           1.43×
1× 4096      45.2      44.3        41.4          32.6           1.39×
4×  512      24.8      25.7        21.5          17.6           1.41×
4× 2048      76.4      76.5        74.8          56.1           1.36×
8× 1024      77.5      77.6        76.7          58.5           1.32×
16× 512      80.6      82.0        79.6          61.5           1.31×
32× 256      87.3      89.4        87.5          67.5           1.29×
```

**v6 全面超越 v3（+3–43%），大 batch 快 29–36%。**

> 注：表中 `v6/v3` 为 **加速比**（v3 耗时 / v6 耗时，>1 表示 v6 更快）。  
> `archive/.../evolution.md` 里同名列曾用 **耗时比**（v6/v3，<1 更快），勿混读。

---

## 2. 早期分析与接口重构

### 2.1 起点：paged_attention → decode 专用（演进报告）

`paged_attention` 沿 FA2：`grid=(query_blocks, nh, batch)`，每 block 处理 `BLOCK_M=8` query × 1 head；8 warp 分 producer/consumer。**decode 每 seq 仅 1 个 query**，8 个 query 槽只有 1 个有效 → query 方向并行浪费，KV 只能串行扫。

**flash_decoding v1** 改为 decode 专用 parallel+reduce：
- parallel：`grid=(tot_block, nh)`，**每 KV block 一 CUDA block**（并行度转到 KV）；初版每 block 1 个 Q head；warp 4–7 预取、0–3 计算。
- reduce：按 seq×head 合并 partial。

### 2.2 算子结构与早期实测（`2026-08-08-flash-decoding-tflop.md`）

- FLOPs：`batch × nh × 4 × totlen × hd`（QK^T + P@V；softmax 不计）。
- 关键改动（同阶段）：grid 按 **nkvh** 派发，每 block 服务 `rep=nh/nkvh=6` 个 Q head → K/V LDG 6×→1×；Q 进寄存器；softmax lane0+shuffle。

早期 warp 并行版 vs FlashInfer（CPU timer，节选）：

| batch×totlen | LLAISYS (us) | FlashInfer (us) | 备注 |
|---:|---:|---:|---|
| 1×256 | 49.1 | 48.3 | 小规模打平 |
| 1×2048 | 75.8 | 53.9 | FI 几乎不随长度涨 |
| 16×2048 | 804.9 | 144.3 | 差 ~5.6× |
| 16×2048 TFLOPS | 0.3 | 1.40 | FI 带宽利用率 ~91% |

三版演进（同算子）：16×2048 旧按 nh 派发 1191.9 → 串行 6head 1091.6 → warp 并行 **804.9**。

**分析要点：**
- decode 是 **memory-bound**；差距主要在同步粒度（每 token `__syncthreads`）、浅双缓冲、独立 reduce、缺少 cp.async/PDL。
- 按 nkvh 派发后小 batch block 数骤降（1×256：48→8），省 LDG 但 occupancy 不足 → 小规模相对旧版可能退化。

旧实现 ncu（batch=4,totlen=2048，按 nh 派发）：SM 74%、DRAM 2.4%、L2 hit 98%、Occupancy 62%、regs/thread 50；smem **写** bank conflict 极高（`copy_4d` float4 布局）。

### 2.3 接口重构 + 2D grid（`2026-08-10-flash-decoding-interface-v2.md`）

- 新增 `tot_block_num`；decode grid `(max_block, nh, batch)` → **`(tot_block_num, nh)`**，用 block_ids 前缀和反推 `(batch_id, local_id)`。
- 模板化 d/dv 分派 `{8,16,32,64,128}`。
- 修复：前缀查找下溢、`block_ids` 2D 索引、小 hd 模板未实例化、`reg_acc` 零长度、`tot_block_num` 越界取末行前缀和。

重构后（best-of-50）：

```
batch×totlen   LLAISYS  FlashInfer   LL/FI
1×256           16.4      40.2       0.41×（快）
1×2048          48.1      41.1       1.17×
16×2048        604.2     111.0       5.4×
```

相对重构前：1×256 -51%，1×2048 -36%，16×2048 -27%。小 batch 2D grid 塞满 SM；大 batch 仍受同步 load + 每 token sync 限制。

---

## 3. v2：GQA HEADS + TILE_K 自动选参

**优化 A：** `grid.y = nh/HEADS`，HEADS∈{1,2,3,6}，同 kv_head 组共享一次 K/V load。  
**优化 B：** `TILE_K`∈{4,8,16,32} 降低 sync 频率。

扫描结论：A 单独 **H3 最优**；B 单独几乎无收益；组合 **H6/T8（大 batch）或 H6/T16**。16×2048：v1 567.5 → 最优 **300.6 us（-47%）**，vs FI 从 5.4×→**2.7×**。

启发式（写入 C wrapper，`heads=0/tile_k=0` 触发）：

```cpp
tokens = tot_block_num * token_num;
heads = tokens<=512 ? 2 : tokens<=2048 ? 3 : 6;
blocks = tot_block_num * (nh / heads);
tile_k = (blocks < 64) ? 16 : 8;
```

全网格 24 config 平均损失 **0.6%**。修复：q 加载条件在 HEADS>consumer 时漏载；`pick_params` 勿重复乘 batch；小 kernel 对比须用 **nsys**（cudaEvent 噪声大）。

---

## 4. v3：cp.async 全体参与

| 版本 | 核心 | 16×2048 |
|------|------|---------|
| v1 | parallel+reduce | ~600 |
| v2 | HEADS+TILE_K+选参 | 337 |
| **v3** | **全体参与 cp.async** | **296** |

取消 producer/consumer：256 线程都发 cp.async → wait → compute。ncu（16×2048 H6T8）：

| 指标 | v2 | v3 分离 | **v3 全体** |
|------|-----|---------|------------|
| barrier | 1.68 | 5.33 | **1.14** |
| scoreboard | 2.29 | 0.78 | **0.83** |

归约：warp0/1 并行分半优于线性/树形（337 vs 342/369 us）。大 batch 相对 v2 约 -6～-11%；小 batch 略差 6–7%。v3 曾接管主接口。

---

## 5. v4：COALESCE（废弃）

动机：物理 TN 64→256 曾降大 batch ~14%；试图用 COALESCE 在不改 TN=64 布局下模拟变长。

COALESCE 单独（H6T8）：**19 个合法 config 全部 CO=1 最优**，无一例外。CO≥2 全面变慢。

原因：物理变长是「单一基址连续 cp.async」；虚拟 COALESCE 每 tile 查指针表，开销抵消 grid 缩减。系统最大 grid 不大（wave 少），缩 grid 几乎无收益。

ncu：C1 vs C4 单 block 微架构指标几乎相同 → 损失来自 wave scheduling，非单 block 效率。**v4 不接入。**

---

## 6. v5：向量化与寄存器压力（废弃）

改动：`float4` 连续寄存器 + `copy_4d` + 向量点积；保留 flatten/COALESCE 模板。912 组合扫描。

- **小 batch（1–2）** 常优于 v3（+6～56%）；**大 batch（≥4）** 常慢于 v3（-2～14%）。
- H3T16 下 COALESCE 首次「有效」；**H6 下 CO>1 仍严重退化**。

ncu（16×2048）：

| | v4 H6T8C1 | v5 H6T8C1 |
|--|-----------|-----------|
| Duration | 395 us | 545 us |
| Occupancy | 31.5% | **16%** |
| Block Limit Regs | 2 | **1** |
| SM Throughput | 61% | **28%** |

根因：float4 推高 regs → 每 SM 只能 1 block → occupancy 减半，向量化 CPI 收益补不回 latency hiding。**v5 不接入**；改进方向是先降寄存器恢复 2 block/SM。

---

## 7. v6：4-warp + 参数最优（自研最优）

数据与扫描结论以 `2026-08-12-flash-decoding-v4v5v6-params.md` 为准；架构动机见 `v6-4warp.md`。

### 7.1 架构改动

相对 v5：`BLOCK_DIM` 256→**128（4 warps）**；`s_q` 用 T(bf16)；2-stage pipeline；大 batch 关键是 **TILE_K=16**（每 warp 4 token 满负荷）。

ncu 路径（16×2048）：v5 545us / occ 16% → v6 ~286us / Block Limit Regs **3** / occ ~24% / SMEM 26KB。再 2-stage：SMEM 22KB，Block Limit SMEM **4**，~265us。

早期「同参对比」曾显示大 batch 慢 v3 ~40%——那是 **TK8** 未匹配 4-warp 负载；全参扫描后用 **H6T16C1S2U1** 反超。

### 7.2 参数空间与 864 组合扫描

| 维度 | 取值 | 说明 |
|------|------|------|
| HEADS | 1,2,3,6 | 每 block 服务 Q head 数（`grid.y=nh/HEADS`） |
| TILE_K | 8,16 | 每次 sync 处理的 token 数 |
| COALESCE | 1,2,4 | 虚拟 KV block 合并数 |
| NUM_STAGES | 2,3 | cp.async 流水线深度（v6 参数化） |
| UNROLL | 0,1 | compute 内层 HEADS 展开 |
| blockDim | 256 / 128 | v5=8-warp / v6=4-warp |

**v6 每 config 最优（par us，9×96=864）：**

| config | v6 最优 |
|--------|---------|
| 1×256 | H1T16C1S2U0: **7.7** |
| 1×1024 | H2T16C1S2U1: **10.2** |
| 1×2048 | H6T16C1S2U1: **14.5** |
| 1×4096 | H6T16C1S2U1: **26.8** |
| 4×512 | H6T16C1S2U1: **14.6** |
| 4×2048 | H6T16C1S2U1: **52.1** |
| 8×1024 | H6T16C1S2U1: **53.6** |
| 16×512 | H6T16C1S2U1: **56.4** |
| 32×256 | H6T16C4S2U1: **59.9** |

维度结论：
1. **TK=16 ≫ TK=8** — 4-warp 时 TK16→每 warp 4 token 满负荷；TK8 每 warp 仅 2 token。**这是大 batch 从慢 40% 翻到快 36% 的关键。**
2. **NUM_STAGES=2 > 3** — 省 4KB smem，Block Limit Shared Mem 3→4。
3. **UNROLL=1 略优**；去掉全部 unroll（含 compute HEADS）会慢 **2.6–3.6×** — compute HEADS 循环必须 unroll。
4. **COALESCE=1** 仍最优（仅 32×256 扫描最优出现 C4，总时间对比仍以 C1 接入为准）。
5. **HEADS**：小 batch（≤1024）用 H1/H2；大 batch 用 H6。

| 版本 | 架构 | 最优（大 batch） | 最优（小 batch） |
|------|------|------------------|------------------|
| v3 | 标量 8-warp 3-stage | H6T8 | H2T16 |
| v4 | 标量 8-warp + COALESCE | H6T8C1 | H2T16C1 |
| v5 | float4 8-warp | H6T8C1 | H3T16C4 |
| v6 | float4 4-warp 2-stage | **H6T16C1S2U1** | **H1T16C1S2U0** |

**v6 为什么赢：** (1) 4-warp+TK16 满负荷；(2) 2-stage → smem 22KB、SMEM 允许 4 blocks/SM（regs 仍限 3）；(3) float4/`copy_4d` 降 LSU（Warp CPI 5.3 vs v4 8.5）。

**接入建议：** 默认 H6T16C1S2U1；`tokens≤512→H1T16`，`≤1024→H2T16`，否则 H6T16；`NUM_STAGES=2`、`UNROLL=1` 固定。

### 7.3 vs FlashInfer：公平口径 vs patch 对照

**主结论（公平口径）：** FlashInfer 的 `DISPATCH_GQA_GROUP_SIZE` 当时不覆盖 **`group_size=6`**，而本仓库主测模型恰是 `nh=12,nkvh=2`。因此 **不能把「未支持该形状 / 或强行 patch 后的 FI」当作默认基线**。  
仅考虑 FlashInfer **现有（原生）已支持的 GQA 配置** 时，自研 v6 **基本上逼近 FlashInfer**——这是对实现水平的正确评价。

早期文档里出现过「v6 全面快于 FI」的表述，对应的是 gs=6 **跑不通或未走专化路径** 时的数字，不可当作 FI 专化 kernel 的性能。反过来，后来 patch 出 gs=6 再比出的 **大 batch ~2–2.6×**，说明的是「若 FI 补上该形状专化实现」的上限差距，**不是**「开箱即用的 FI 碾压自研」。

**补充：patch `group_size==6` 后同环境 nsys**（研究对照；修改 `utils.cuh` 的 `DISPATCH_GQA_GROUP_SIZE`，清 JIT，`CC=gcc-12 CXX=g++-12`；par+reduce vs 单 kernel）：

```
b×totlen    v6(us)    FlashInfer(us)   v6/FI
1×  256       9.5        10.0          1.05×  （v6 略快）
1× 1024      12.7        10.0          0.79×
1× 2048      18.0        12.3          0.68×
1× 4096      32.5        15.9          0.49×
4×  512      17.5        12.3          0.70×
4× 2048      56.1        25.5          0.45×
8× 1024      58.4        25.6          0.44×
16× 512      61.7        25.6          0.42×
32× 256      67.6        25.4          0.38×
```

在该 **补丁后** 形状上，大 batch FI 专化核更快；ncu（16×512）可解释差距来源：

| 指标 | v6 H6T16C1S2U1 | FlashInfer（patch gs=6） |
|------|----------------|--------------------------|
| Registers/Thread | **159** | **62** |
| Achieved Occupancy | 25% | **62.5%** |
| Blocks/SM | 3 | **10** |
| SM Throughput | 49% | **61.6%** |
| Stall Long Scoreboard | 1.44 | **0.21** |
| Stall Wait | 1.07 | 0.64 |
| DRAM Throughput | 100% | 100% |
| Duration | 61.7 us | 25.6 us |

补丁后 FI：`grid=(batch×partition, nkvh)`，`block=(16,6)`≈96 threads，~7KB smem → 高 occupancy。v6：128 threads / 4 warps，159 regs，22KB smem → 3 blocks/SM。两者 DRAM 都报 100% 时 v6 仍慢，说明瞬时峰值带宽 ≠ 有效带宽；reduce partial 约多 ~10% 流量。

**怎么读这两套对比：**  
- **评价实现是否够用：** 看 FI **原生支持** 的配置 → v6 基本逼近。  
- **若继续抠 Qwen gs=6 专化上限：** 看 patch 表 + occupancy/regs → 需更低寄存器或 partition-kv 级架构。

---

## 8. v7/v8/v9：逼近 FlashInfer 的失败路径

| 版本 | 思路 | 结果 | 关键教训 |
|------|------|------|---------|
| **v7** | 在 v6 上换 FI 式 `(16,HEADS)` 布局 | 全面慢于 v6 | block-per-KV-block 上 shuffle+smem 冗余读不兼容 |
| **v8** | block-per-seq×nkvh 单 kernel；先无 smem，再加 cp.async | 直接读慢 3–90×；+pipeline 仍慢 v6 2–50× | regs/occ/延迟可改善，但 **grid=batch×nkvh 太小**；需 partition-kv |
| **v9** | v6 grid + v8 线程布局 | 仍慢于 v6 | 固定开销使 regs 停在 114，shuffle 暴露 |

三层矛盾：

| 架构 | 优势 | 瓶颈 |
|------|------|------|
| block-per-KV-block（v6） | grid 大 | regs 159 → occ 25% |
| block-per-seq（v8） | regs~72，occ~56% | grid 小 |
| 融合（v9） | 折中 | shuffle + 冗余读 + regs 114 |

**逼近 FI（补丁后 gs=6 上限）：** 保留 v6 大 grid、压 regs；或完整 partition-kv。去掉 reduce 最多省 ~10%，解释不了补丁后的 ~2×。  
**公平评价：** 仍以 FI 原生已支持配置为准——v6 已基本逼近。

---

## 9. 与 FlashInfer 最终对比与结论

1. **优化主线：** 消 LDG 冗余（v2）→ 藏访存（v3 cp.async）→ 降 regs/提 occ + 参优（v6：4-warp、2-stage、TK16、864 扫描）。  
2. **自研最优 = v6 H6T16C1S2U1**（大 batch 相对 v3 约 +30%）。  
3. **vs FlashInfer（主结论）：** FI **未原生提供 `group_size=6`**；在其 **现有已支持配置** 上，v6 **基本逼近 FI**。  
4. **vs FlashInfer（补充）：** 人为 patch gs=6 后，大 batch 仍可差 ~2–2.6×（regs/occupancy）；这是专化上限分析，不是开箱对比。  
5. **废弃：** v4 COALESCE、v5 大 batch、v7–v9。  
6. **测量：** 小 kernel 用 nsys；引用 FI 数字时必须写明是否 patch、是否属 FI 原生支持形状。

---

## 10. 原文索引

完整原文（含复跑脚本路径）见：

`docs/optimization/archive/flash-decoding/`

| 文件 | 主题 |
|------|------|
| `2026-08-08-flash-decoding-tflop.md` | 早期结构、TFLOPS、ncu、优化方向 |
| `2026-08-10-flash-decoding-interface-v2.md` | 2D grid / tot_block_num / bugfix |
| `2026-08-10-flash-decoding-param-autotune.md` | HEADS×TILE_K 与启发式 |
| `2026-08-10-flash-decoding-v3-cpasync.md` | v3 全体参与 |
| `2026-08-12-flash-decoding-v4-coalesce.md` | COALESCE 全面实验 |
| `2026-08-12-flash-decoding-v5-vectorize.md` | float4 / 912 扫描 |
| `2026-08-12-flash-decoding-ncu-analysis.md` | v4/v5 ncu |
| `2026-08-12-flash-decoding-v6-4warp.md` | 4-warp / stages / unroll |
| `2026-08-12-flash-decoding-v4v5v6-params.md` | 864 扫描与 FI 同环境对比 |
| `2026-08-12-flash-decoding-v7-threadlayout.md` | FI 布局移植失败 |
| `2026-08-12-flash-decoding-v8-blockperseq.md` | block-per-seq 与三层瓶颈 |
| `2026-08-12-flash-decoding-v9-merge.md` | v6+v8 融合失败 |
| `2026-08-13-flash-decoding-evolution.md` | 演进总报告 |
