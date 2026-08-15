# Flash Decoding 优化汇总

> 合并自多份 `2026-08-08`～`2026-08-13` flash-decoding 专题报告；以演进报告为主干，附录保留各版关键数据表与结论。
>
> 硬件：RTX 4060 Laptop（24 SM, 256 GB/s, Ada CC 8.9）  
> 模型：DeepSeek-R1-Distill-Qwen-1.5B（NH=12 / NKVH=2 / HD=128）  
> 系统约束：TN=64，BATCH_MAX_TOKEN_NUM=8192，MAX_TOKEN_NUM=2048

## 目录

1. [演进总览与主线分析](#演进主线)
2. [附录：分版本关键数据](#附录分版本关键数据)

<a id="演进主线"></a>

# 演进主线

日期: 2026-08-13
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）
模型: DeepSeek-R1-Distill-Qwen-1.5B, NH=12/NKVH=2/HD=128
系统约束: TN=64, BATCH_MAX_TOKEN_NUM=8192, MAX_TOKEN_NUM=2048

## 演进总览 (16×2048)

| 版本 | 核心优化 | 16×2048 (us) | vs FlashInfer | 状态 |
|------|---------|-------------|---------------|------|
| paged_attention | FA2 风格, query 方向并行 | ~800+ | — | decode 并行度浪费 |
| v1 (初版) | 并行度转 KV 方向, parallel+reduce, 按 nh | ~600 | 5.4× | 基线 |
| v2 | GQA HEADS 共享 + TILE_K + 自动选参 | 327 | 2.7× | 架构确立 |
| v3 | cp.async 全体参与流水线 | 296 | 2.4× | 主接口 |
| v4 | COALESCE 虚拟变长 KV block | 无效 | — | 实验 |
| v5 | float4 向量化 + 连续寄存器 | 大 batch 退化 | — | 实验 |
| v6 | 4-warp + 2-stage + TK16 全参数最优 | 快 v3 30% | — | **自研最优** |

注: 16×2048 的 v1-v3 来自 v3 优化文档; v6 在 16×512/4×2048 等 config 全面优于 v3。
v2 数据 (327us) 为文档值 (cudaEvent 测量), v3 296us 为 nsys。

## 第一阶段: paged_attention → flash_decoding

### 起点: paged_attention (FA2 风格, 服务 prefill + decode)

`paged_attention_kernel` 以 flash-attention-v2 为基础:
- **grid = (query_blocks, nh, batch)**: blockIdx.z=batch, blockIdx.y=head,
  blockIdx.x=query 分块。每 block 处理 **BLOCK_M=8 个 query × 1 个 head**
- 256 threads = 8 warps: **4 consumer (计算) + 4 producer (加载 KV) 分离**
- KV_TILE=4 双缓冲, 每 KV_TILE 个 token 一次 `__syncthreads()`
- online softmax (running max/sum), float4 向量化
- 服务 prefill 和 decode 两阶段 (is_prefill 分支)

**decode 瓶颈**: decode 每 seq 只有 1 个 query token, 而 paged_attention 按
query 方向并行 (BLOCK_M=8)。decode 时 8 个 query 槽只有 1 个有效 → **query 方向
并行度大量浪费**, KV 方向只串行遍历。

### flash_decoding v1 (decode 专用, 并行度转 KV 方向)

为 decode 单独设计 parallel + reduce 两 kernel:
- **parallel**: grid = (tot_block, nh), **每 KV block 一个 block** (并行度
  从 query 方向转到 KV block 方向)。**每 block 只处理 1 个 query head**,
  batch 通过 block_ids 前缀和映射 (table_id → batch_id)。
  256 threads = 8 warps: **warp 4-7 预取 K/V (双缓冲 4 槽), warp 0-3 计算**,
  producer/consumer 分离。算该 KV block 的 attention partial (attn_acc/sum/max),
  写全局 workspace。
- **reduce**: 每 block 合并一个 seq×head 的所有 KV block partial, 写最终 attn_val

初版 16×2048 ≈ 600us, 比 FlashInfer 慢 5.4×。

## 第二阶段: v1 → v2 (架构确立)

### v1 的瓶颈

grid.y = nh = 12, 每 block 只处理 1 个 query head。同 kv_head 的 6 个 head-block
(如 nh=0..5 都映射 nkvh=0) **各读一次同一份 K/V** → 6× LDG 冗余。

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

## 各版本统一实测数据 (v3-v6 同 nsys 方法, 各版本最优参数)

v1/v2 数据来自原优化文档 (cudaEvent 测量, 与 nsys 略有差异但趋势一致):

```
b×totlen    v1      v2     v3(us)  v4(us)  v5(us)  v6(us)  v6/v3
───────────────────────────────────────────────────────────────────
1×  256      —      29.7    11.8    13.2    11.6    11.5   0.97x
1× 1024      —      36.1    16.6    17.6    21.1    14.8   0.89x
1× 2048      —      48.2    25.7    28.2    31.1    18.1   0.70x
1× 4096      —       —      45.1    48.1    59.2    32.6   0.72x
4×  512      —       —      24.8    28.4    31.1    17.5   0.70x
4× 2048      —      97.9    76.3    83.3   105.2    56.7   0.74x
8× 1024      —     100.7    77.4    85.6   108.5    58.6   0.76x
16×  512     —       —      80.6    90.3   114.2    62.1   0.77x
32×  256     —     110.3    87.3    98.7   125.9    67.1   0.77x
16× 2048     ~600   327.2   296.5     —       —       —    —
```

注:
- v1/v2 来自 `2026-08-10-flash-decoding-v3-cpasync.md` (v2 是 v3 文档的基线)
- v3/v4/v5/v6 为本次统一 nsys 实测 (各版本最优参数):
  v3=H6T8, v4=H6T8C1, v5=H6T8C1, v6=H6T16C1S2U1
- v5 大 batch 退化 (105 vs v3 76) 因 float4 寄存器压力 → occupancy 减半

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

**性能** (v5 H6T8C1 统一实测 vs v3):

| config | v3(us) | v5(us) | v5/v3 |
|--------|--------|--------|-------|
| 1×256 | 11.8 | 11.6 | 0.98x (略快) |
| 1×2048 | 25.7 | 31.1 | 1.21x 慢 |
| 4×2048 | 76.3 | 105.2 | **1.38x 慢** |
| 8×1024 | 77.4 | 108.5 | **1.40x 慢** |
| 32×256 | 87.3 | 125.9 | **1.44x 慢** |

**H6 时大 batch 严重退化** (慢 v3 21-44%)。float4 acc[6] 使寄存器 >192,
occupancy 减半 (16% vs 31%), 向量化收益 < occupancy 损失。

v5 早期测试在小 batch H2T16 场景有 +11-24% 收益 (低 HEADS 寄存器充裕),
但 H6 大 batch 是主场景, v5 不接入。

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

**性能对比** (v6 H6T16C1S2U1 vs v3 H6T8, 统一实测):

```
b×totlen    v3(us)   v6(us)   v6/v3
────────────────────────────────────
1×  256     11.8    11.5    0.97x
1× 1024     16.6    14.8    0.89x
1× 2048     25.7    18.1    0.70x
1× 4096     45.1    32.6    0.72x
4×  512     24.8    17.5    0.70x
4× 2048     76.3    56.7    0.74x
8× 1024     77.4    58.6    0.76x
16× 512     80.6    62.1    0.77x
32× 256     87.3    67.1    0.77x
```

**v6 全面超越 v3 (大 batch 快 24-30%, 小 batch 快 3-11%)**。
v6/v3 < 1 表示 v6 更快 (耗时更短)。

## 与 FlashInfer 的最终对比 (patch group_size=6 后同环境实测)

```
b×totlen    v6(us)   FI(us)   v6/FI
───────────────────────────────────
1×  256     11.5    10.0    1.05x (v6 略快)
1× 2048     18.1    12.3    0.68x
1× 4096     32.6    15.9    0.49x
4× 2048     56.7    25.5    0.45x
16× 512     62.1    25.7    0.42x
32× 256     67.1    25.5    0.38x
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
   提高 occupancy + 参数最优 (v6: 4-warp + 2-stage + TK16 + 864 组合全扫描)
2. **v6 是自研最优**: 统一实测大 batch 快 v3 24-30% (4×2048: 76.3→56.7us),
   小 batch 快 3-11%。达 FlashInfer 的 45-68% (大 batch)。
3. **unroll 必须保留**: 去掉 compute unroll 慢 2.6-3.6x
4. **与 FlashInfer 差距**: ~2.2x (大 batch) 来自 occupancy (寄存器 159 vs 62),
   需架构级重写 (block-per-seq + partition-kv)

## 数据来源与测量方法

- v1: `2026-08-08-flash-decoding-tflop.md` (16×2048 ≈ 600us)
- v2: `2026-08-10-flash-decoding-v3-cpasync.md` 基线 (cudaEvent 测量)
- v3/v4/v5/v6: `test/profile/_bench_evolution.py` 统一 nsys 实测
  (同方法同 config 集, 各版本最优参数: v3=H6T8, v4=H6T8C1, v5=H6T8C1, v6=H6T16C1S2U1)
- FlashInfer: patch group_size=6 后同环境 nsys 实测

## v6 reduce kernel 占比 (H6T16C1S2U1 统一实测)

**结论**: reduce kernel 占比小 batch 高 (20%), 大 batch 低 (8-11%)。
v6 与 FlashInfer 的 2.2x 差距主要不是 reduce 开销, 而是 parallel kernel 的
occupancy (159 regs → 25%) 导致有效带宽低。去掉 reduce 最多省 ~10%。

---

<a id="附录分版本关键数据"></a>

# 附录：分版本关键数据

下列内容从各专题报告抽取标题、表格、数据块与结论句，去掉重复叙述。

## 来源：`2026-08-08-flash-decoding-tflop.md`

**原题：** flash-decoding 算子性能分析

## 测试配置
## 代码结构
### parallel kernel
| warp 0–5 (consumer) | warp 6 (producer K) | warp 7 (producer V) |
|---|---|---|
| 各处理 1 个 query head（`nh = (nh/nkvh)*nkvh + warp_id`） | 从 global 预取下一 token 的 K 到 smem | 从 global 预取下一 token 的 V 到 smem |
| Q 常驻寄存器 `reg_q[4]`，逐 token 做点积 + online softmax + 累加 | 双缓冲 `now ^= 1` | 双缓冲 `now ^= 1` |

### reduce kernel
每个 block 按 nh 派发（grid.y = nh = 12），每个 block 合并一条序列一个 head 的所有块结果。256 线程 8 warp 双缓冲：warp 4–7 预取下一块的 acc/sum/max，warp 0–3 做 online softmax merge。最终 warp 0 写出 `attn_val: (batch, nh, hd)`。
### 与旧实现的关键差异
| | 旧实现（按 nh 派发） | 新实现（按 nkvh 派发） |
|---|---|---|
| 网格 | grid.y = nh = 12，每 block 1 个 head | grid.y = nkvh = 2，每 block 6 个 head（warp 并行） |
| K/V 读取 | 同份 KV 被 6 个 head-block 各读一次（6× LDG） | 一份 KV 进 smem 服务 6 个 head（1× LDG） |
| Q 存放 | 放 smem `s_q`（6×d 空间） | 放寄存器 `reg_q[4]`（省 smem） |
| softmax | 32 lane 各算一遍 expf | lane 0 单算 + shuffle 广播 |
| 双缓冲 | 4 槽 `now ^= 4` | 2 槽 `now ^= 1`（每 token 一槽） |

## 性能对比
### 延迟和 TFLOPS
| batch × totlen | LLAISYS (us) | FlashInfer (us) | LLAISYS TFLOPS | FlashInfer TFLOPS |
|---:|---:|---:|---:|---:|
| 1 × 256 | 49.1 | 48.3 | 0.1 | 0.03 |
| 1 × 512 | 49.2 | 55.0 | 0.1 | 0.06 |
| 1 × 1024 | 60.4 | 48.8 | 0.1 | 0.13 |
| 1 × 2048 | 75.8 | 53.9 | 0.2 | 0.23 |
| 4 × 256 | 59.4 | 49.6 | 0.1 | 0.13 |
| 4 × 512 | 74.5 | 53.6 | 0.2 | 0.23 |
| 4 × 1024 | 131.1 | 58.0 | 0.2 | 0.43 |
| 4 × 2048 | 222.2 | 68.1 | 0.2 | 0.74 |
| 8 × 256 | 90.1 | 52.5 | 0.2 | 0.24 |
| 8 × 512 | 134.0 | 56.2 | 0.2 | 0.45 |
| 8 × 1024 | 222.2 | 68.4 | 0.2 | 0.74 |
| 8 × 2048 | 420.9 | 90.3 | 0.2 | 1.11 |
| 16 × 256 | 134.1 | 60.5 | 0.2 | 0.42 |
| 16 × 512 | 224.3 | 72.8 | 0.2 | 0.69 |
| 16 × 1024 | 422.9 | 90.1 | 0.2 | 1.12 |
| 16 × 2048 | 804.9 | 144.3 | 0.3 | 1.40 |

### 三版演进对比（同一算子）
| config | 旧：按 nh 派发 | 串行 6head | warp 并行（当前） |
|---|---:|---:|---:|
| 1 × 256 | 25.6 | 69.7 | 49.1 |
| 1 × 2048 | 89.1 | 102.2 | 75.8 |
| 8 × 2048 | 602.1 | 563.2 | 420.9 |
| 16 × 2048 | 1191.9 | 1091.6 | 804.9 |

从旧版到 warp 并行版，大 batch 提速 ~30%（16×2048: 1192→805us）。但中间那版"按 nkvh 派发 + 每 warp 串行算 6 个 head"反而比旧版慢（1×256: 69.7 vs 25.6）——原因在最后说明。
### 几个直接能看出来的
**TFLOPS 天花板。** LLAISYS 在 0.1–0.3，FlashInfer 到 1.40（16×2048 时带宽利用率 91%，233 GB/s）。decode 注意力是 memory-bound，0.3 TFLOPS 不是算得慢，是数据供不上。FlashInfer 供得上的原因是单 kernel + 高占用流水。
### 为什么 batch=1 反而退化（相对旧版）
## 与 FlashInfer 的剩余差距
| | LLAISYS（当前） | FlashInfer |
|---|---|---|
| 网格派发 | 按 nkvh=2 ✓ 已对齐 | 按 nkvh |
| K/V 读取 | 1× ✓ 已对齐 | 1×（cp.async 异步加载） |
| softmax | lane0 单算 + shuffle ✓ 已对齐 | lane0 + exp2 |
| 收尾 | 独立 reduce kernel（一次额外 launch + workspace 读写） | 单 kernel 内完成（partition-kv + merge） |
| K/V 加载 | `copy_4d`（同步 LDG → smem） | `cp.async`（异步拷贝，与计算重叠） |
| 流水深度 | 2 槽双缓冲，每 token 同步一次 | 多级 smem tile 流水，整块预取 |
| 同步粒度 | 每 token 一次 `__syncthreads()` | 每 tile 一次 `block.sync()` |

## ncu profiling 结果（旧实现）
### 总体指标
| 指标 | 值 | 说明 |
|---|---|---|
| SM Throughput | 74% | SM 计算管线忙 |
| L1/TEX Throughput | 72% | L1/TEX 管线也忙 |
| DRAM Throughput | 2.4% | 数据几乎不从 DRAM 来 |
| L2 Hit Rate | 98% | 数据基本从 L2 读 |
| Occupancy | 62% | 每个 SM 同时跑 4–5 block |
| Registers/thread | 50 | |
| Shared Memory/block | 10.78 KB | |

### Stall 原因分解
| Stall 原因 | cycles | 对应代码 |
|---|---|---|
| Wait | 1.56 | expf 长延迟指令的依赖链 |
| Not Selected | 1.56 | warp 准备好但 scheduler 选了别的 |
| Long Scoreboard | 1.16 | 等 LDG 数据（L2 命中但仍有延迟） |
| Short Scoreboard | 0.78 | 等 smem 读写 |
| Math Pipe | 0.78 | expf/sinf/cosf 占满计算管线 |

### 共享内存 bank conflict
| | bank_conflicts |
|---|---|
| 读 (ld) | 0 |
| 写 (st) | 2,359,296 |

## 优化方向（按收益）
## 复跑命令
```bash
# LLAISYS 基准
wsl -e bash -c "cd /home/songjq/llaisys-main && ./venv/bin/python \
  test/benchmark/ops/benchmark_flash_decoding.py --dtype bf16"

# FlashInfer 基线
wsl -e bash -c "cd /home/songjq/llaisys-main && \
  CUDA_HOME=/usr/local/cuda-13.1 CC=gcc-12 CXX=g++-12 \
  ./venv/bin/python test/benchmark/ops/flashinfer_decode.py"

# per-head 数值校验（对比 torch 参考，maxdiff < 0.01 为正常）
wsl -e bash -c "cd /home/songjq/llaisys-main && ./venv/bin/python test/profile/_debug_flash_decoding.py"

# nsys 看 kernel 时间分配
nsys profile -o /tmp/fd --force-overwrite true \
  /home/songjq/llaisys-main/venv/bin/python test/profile/_profile_flash_decoding.py

# ncu 看 parallel kernel 内部
/usr/local/cuda/bin/ncu --set detailed --launch-skip 0 --launch-count 1 \
  -k 'regex:flash_decoding_parallel.*' \
  /home/songjq/llaisys-main/venv/bin/python test/profile/_ncu_flash_decoding.py
```

## 来源：`2026-08-10-flash-decoding-interface-v2.md`

**原题：** flash-decoding 接口重构 + 性能对比

## 接口变更（2026-08-10）
### 修复的 bug
## 测试配置
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
## 验证
- `test/ops/flash_decoding.py --device nvidia` PASS（含单样本 + batched + 多 dtype）
- `test/ops/paged_attention.py --device nvidia` PASS（prefill 路径回归）
- `test/test_runtime.py --device nvidia` PASS
- benchmark 正确性 PASS（per-head maxdiff < 容差）

## 来源：`2026-08-10-flash-decoding-param-autotune.md`

**原题：** flash-decoding 参数自动调优

## 两个正交优化点
### 优化A: grid.y = nh / HEADS（GQA head 共享）
| HEADS | grid.y | 每 block head 数 | K/V load 次数（nh=12） | 含义 |
|-------|--------|-----------------|----------------------|------|
| 1 | 12 | 1 | 12 | v1 基线 |
| 2 | 6 | 2 | 6 | 成对共享 |
| 3 | 4 | 3 | 4 | 3-head 共享 |
| 6 | 2 | 6 | 2 | 完全 GQA 共享（= FlashInfer grid.y=nkvh） |

### 优化B: 双缓冲粒度 TILE_K（每次 sync 的 token 数）
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

## 全网格 nsys 扫描 + 启发式自动选参
### 规律
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
## 修复的 bug
3. **测量方法**：cudaEvent 批量计时对 batch=1 小 kernel 噪声大（曾显示 1×1024 损失 42%），nsys 精确测量后同 config 仅 -0.2%。**结论：小 kernel 性能对比必须用 nsys**。
## 验证
- 全部 16 个 (HEADS, TILE_K) 组合正确性 PASS（max_diff < 0.005）

## 来源：`2026-08-10-flash-decoding-v3-cpasync.md`

**原题：** flash-decoding v3: cp.async 全体参与流水线

## 版本演进
| 版本 | 核心 | 16×2048 | 说明 |
|------|------|---------|------|
| v1 | 原始并行+reduce | ~600us | 用户初版 |
| v2 | GQA HEADS 共享 + TILE_K 双缓冲 + 自动选参 | 337us | 同步 load，producer/consumer 分离 |
| v3 | **cp.async 全体参与流水线** | **296us** | 本版 |

## v3 架构
## 为什么取消 producer/consumer
→ sync 处 producer 慢、consumer 空等 → 严重不平衡。
| 指标 | v2 | v3 分离 | **v3 全体** |
|------|-----|---------|------------|
| barrier | 1.68 | 5.33 | **1.14** |
| scoreboard (L2 load) | 2.29 | 0.78 | **0.83** |

## 归约优化实验（ncu 精确对比，16×2048）
| 方案 | 关键路径 | 耗时 | barrier |
|------|---------|------|---------|
| 线性（warp0 串行并 1..7） | 7 次 | 342us | 1.14 |
| 树形（stride 1→2→4，3 轮） | 3 轮 | 369us | 1.41 |
| **warp0/1（并行分半）** | 3 并行 + 1 | **337us** | 1.25 |

- **warp0/1 最优**：warp0 并 2-4，warp1 并 5-7（并行 3 次），最后 warp0 并 warp1
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

## 最终主接口性能（v3 接管）
## 接入状态
- 正确性: flash_decoding / paged_attention / runtime 测试全 PASS

## 来源：`2026-08-12-flash-decoding-v4-coalesce.md`

**原题：** flash_decoding v4: COALESCE 虚拟变长 KV block 实验

## 系统配置
```cpp
// src/config.hpp
KV_CACHE_TOKEN_NUM = 64    // KV cache 块大小 (token)
KV_CACHE_BLOCK_NUM = 128   // KV cache 总块数
BATCH_MAX_TOKEN_NUM = 8192 // 批次最大并行 token 总数
MAX_TOKEN_NUM = 2048       // 序列最大长度上界
MAX_BLOCK_NUM = 33         // 单序列最大块数 (32 + 1 前缀列)
```

## 动机
## 设计迭代
| 版本 | 方法 | 16×2048 H6T8 CO=1→CO=4 |
|------|------|------------------------|
| v4a | 每 block 独立 pipeline + outer loop | 274→393us (+43%) |
| v4b | flatten tile 迭代 + 内联 cp_load | 627→542us |
| v4c | flatten tile + 预计算 `k_blk[COALESCE]` 指针数组 | 324→333us (+3%) |

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
### 全部参数组合 vs v3 基线
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

H=3/TK=16/CO=8 从未进入 top-1。切换到 H=3 本身已比 H=6 慢 8-120%，
### 与 FlashInfer 对比
| batch×totlen | LLAISYS(us) | FlashInfer(us) | LL/FI |
|-------------|-------------|-----------------|-------|
| 1 × 256 | 7.7 | 48.3 | 0.16× |
| 1 × 2048 | 25.9 | 53.9 | 0.48× |
| 4 × 2048 | 76.5 | 68.1 | 1.12× |
| 8 × 1024 | 77.6 | 68.4 | 1.13× |
| 16 × 256 | 46.0 | 60.5 | 0.76× |

### 带宽
## 为什么 COALESCE 无效
## 结论
1. **COALESCE 在本系统约束下无效。** 19 合法 config 中 H=6/TK=8 控制组显示 CO>1 无一胜出。
4. **代码审查结论：kernel 逻辑无 bug。** 任务映射、指针预计算、flatten tile 迭代、
## 测试脚本

## 来源：`2026-08-12-flash-decoding-v5-vectorize.md`

**原题：** flash_decoding v5: flatten pipeline + 向量化访存 (基于 v4 结构)

## 改动 (vs v4)
| 组件 | v4 | v5 |
|------|-----|-----|
| 寄存器布局 | `reg_acc[h][(HD+31)>>5]` strided | `float4 reg_acc[h]` 连续 (lane i→元素 i×4..i×4+3) |
| Q→smem | 标量 `cast<float>(qp[col])` per element | `copy_4d` 128-bit bf16→float4 |
| K/V smem→reg | 标量 `cast<float>(kt_[c])` per element | `copy_4d` ushort4→float4, 一次 4 元素 |
| 点积 | strided 循环 | `dot = q.x*k.x + q.y*k.y + q.z*k.z + q.w*k.w` |
| 归约 merge | strided 标量 | `float4` 逐元素操作 |

## 全参数扫描 (912 组合)
### v5 最优 vs v3 基线
```
b×totlen      v3(H6T8)  v5_best  v5/v3  best(H,TK,CO)
─────────────────────────────────────────────────────────
 1 ×  256       11.4       7.3   0.64x  H=1 TK=16 CO= 1
 1 ×  512       11.2       8.5   0.76x  H=2 TK=16 CO= 1
 1 × 1024       15.4      14.5   0.94x  H=2 TK=16 CO= 1
 1 × 2048       23.5      22.8   0.97x  H=3 TK=16 CO= 3
 1 × 4096       41.4      41.4   1.00x  H=3 TK=16 CO= 6
 2 ×  256       11.1       8.6   0.77x  H=2 TK=16 CO= 1
 2 ×  512       15.7      14.5   0.92x  H=2 TK=16 CO= 1
 2 × 1024       25.3      22.9   0.90x  H=3 TK=16 CO= 3
 2 × 2048       39.5      41.4   1.05x  H=3 TK=16 CO= 6
 4 ×  256       16.2      15.3   0.95x  H=2 TK=16 CO= 2
 4 ×  512       25.7      23.4   0.91x  H=3 TK=16 CO= 3
 4 × 1024       39.5      42.0   1.06x  H=3 TK=16 CO= 6
 4 × 2048       70.0      78.7   1.12x  H=3 TK=16 CO= 4
 8 ×  256       27.2      26.2   0.96x  H=3 TK=16 CO= 1
 8 ×  512       42.9      43.9   1.02x  H=3 TK=16 CO= 2
 8 × 1024       70.7      80.7   1.14x  H=3 TK=16 CO= 2
16 ×  256       46.0      48.5   1.05x  H=3 TK=16 CO= 2
16 ×  512       73.8      83.5   1.13x  H=3 TK=16 CO= 4
32 ×  256       80.5      91.6   1.14x  H=3 TK=16 CO= 4
```

**v5 在小 batch (1-2) 全面优于 v3 (+6-56%)，在大 batch (≥4) 全面慢于 v3 (-2-14%)。**
### COALESCE 效果 (H=3, TK=16 — v5 最常见最优参数组)
```
b×totlen      CO=1    CO=2    CO=3    CO=4    CO=6    CO=8   best
──────────────────────────────────────────────────────────────────
 1 × 2048      24.6    26.6    22.8    28.7    33.8    43.8   CO=3
 2 × 2048      43.8    41.7    42.0    52.2    41.4    52.8   CO=6
 4 ×  512      24.6    29.2    23.4    28.9    40.7    44.4   CO=3
 4 × 2048      81.5    78.9    79.8    78.7    80.5   100.5   CO=4
 8 × 1024      84.3    80.7    83.0    80.8    80.8   101.1   CO=2
16 ×  512      90.3    85.5    89.0    83.5    96.7   103.3   CO=4
32 ×  256     104.2    95.5   106.6    91.6    97.9    92.0   CO=4
```

config 优于 CO=1。最优 CO 随 totlen 增大而增大——小 totlen 用 CO=3，大 totlen 用 CO=4-6。
### H=6,TK=8 中 COALESCE 依然无效
```
b×totlen      CO=1    CO=2    CO=3    CO=4    CO=6    CO=8
───────────────────────────────────────────────────────────
 4 × 2048      96.4   137.9   136.4   134.9   136.9   176.1
 8 × 1024      98.8   140.1   139.9   136.5   138.9   177.1
16 ×  512     104.2   143.7   144.0   139.8   162.4   180.3
```

H=6 时 CO>1 全部严重退化 (CO=2 慢 38-44%)。与 v4 结论一致——高 HEADS 下寄存器压力
### 按 batch 的最优参数规律
| batch | 小 totlen (256-512) | 大 totlen (2048-4096) |
|-------|--------------------|-----------------------|
| 1-2 | H=1-2, TK=16, CO=1 | H=3, TK=16, CO=3-6 |
| 4-8 | H=2-3, TK=16, CO=1-3 | H=3, TK=16, CO=2-4 |
| 16-32 | H=3, TK=16, CO=2 | H=3, TK=16, CO=4 |

## vs v3 综合对比
| 场景 | 结论 |
|------|------|
| batch=1-2, totlen≤1024 | v5 显著优于 v3 (+6-56%) |
| batch=1-2, totlen≥2048 | v5 追平或略优于 v3 (±5%) |
| batch≥4, totlen≤512 | v5 追平 v3 (±5%) |
| batch≥4, totlen≥1024 | v3 优于 v5 (+2-14%) |

## 结论
   但最优仍不如 v3(H6T8)
3. **H=6 下 COALESCE 始终无效** — 寄存器压力根因未改变
4. **v5 不接入主接口**，保留为实验参考。主接口维持 v3
## 测试脚本

## 来源：`2026-08-12-flash-decoding-ncu-analysis.md`

**原题：** flash_decoding v4 & v5 ncu 性能分析

## 关键指标对比
```
指标                          v4_H6T8C1    v4_H6T8C4    v5_H6T8C1    v5_H3T16C4
──────────────────────────────────────────────────────────────────────────────
Duration (us)                   395.1        395.7        545.0        546.1
Elapsed Cycles                 758,515      759,655     1,046,232    1,048,442
Grid Size                      1024          1024         1024         1024
Block Size                      256           256          256          256
──────────────────────────────────────────────────────────────────────────────
Achieved Occupancy             31.51%       31.56%       15.98%       16.01%
Theoretical Occupancy          33.33%       33.33%       16.67%       16.67%
Waves Per SM                    21.33        21.33        42.67        42.67
──────────────────────────────────────────────────────────────────────────────
Block Limit - Registers           2            2            1            1
Block Limit - Shared Mem          2            2            1            1
Block Limit - SM                 24           24           24           24
Block Limit - Warps               6            6            6            6
Dynamic SMEM Per Block (KB)    40.32        40.32        40.32        40.32
──────────────────────────────────────────────────────────────────────────────
Compute (SM) Throughput        60.98%       60.89%       28.35%       28.29%
Memory Throughput              60.98%       60.89%       25.05%       25.02%
DRAM Throughput                34.57%       34.41%       25.05%       25.02%
L1/TEX Hit Rate                 8.86%        8.84%       12.06%       12.05%
L2 Hit Rate                    13.75%       13.76%       13.34%       13.26%
Warp Cycles Per Issued Instr    8.50         8.50         6.64         6.64
──────────────────────────────────────────────────────────────────────────────
Shared Memory Spilling            0            0            0            0
Avg. Divergent Branches           0            0            0            0
```

## 分析
### 1. v4: COALESCE (C1 vs C4) 对微架构无影响
### 2. v5 vs v4: 寄存器压力导致 occupancy 减半——这是核心瓶颈
| 根因指标 | v4 | v5 |
|---------|-----|-----|
| Block Limit Registers | **2** | **1** |
| Achieved Occupancy | **31.5%** | **16.0%** |
| Waves Per SM | 21.33 | **42.67** (2×) |

### 3. v5 H6T8C1 vs v5 H3T16C4: 寄存器压力相似，均受 Block Limit 限制
在 16×512 配置下快 25%。差异来自 TILE_K=16 减少 sync 频率和 H=3 减少 Q load + 点积次数，
### 4. L1/L2 Cache
## 改进方向
| 优先级 | 方向 | 预期收益 | 依据 |
|--------|------|---------|------|
| **P0** | **降低寄存器压力** | 恢复 2 block/SM (+100% occupancy) | Block Limit Registers: 1→2 是根因 |
| P0a | `reg_acc` 放回 smem (HEADS≥3 时) | 省 24 float/thread (H=6) | v3 用 smem 存 acc 时 occupancy=62% |
| P0b | 减少 float4 临时寄存器 | 省 10-15 regs | copy_4d 展开 + 点积累加 |
| P1 | 异步 cp.async 与计算重叠加深 | 5-10% | 当前 3-stage, FlashInfer 用更深流水 |
| P2 | COALESCE 仅在大 grid (≥512 blocks) 启用 | 0-5% | 仅在 wave 数足够大时有效 |

## 测试脚本

## 来源：`2026-08-12-flash-decoding-v4v5v6-params.md`

**原题：** flash_decoding v4/v5/v6 各版本最优配置扫描

# flash_decoding v4/v5/v6 各版本最优配置扫描
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
   并行度不足。**这是 v6 大 batch 从慢 40% 到快 36% 的关键。**
4. **COALESCE=1 最优** — CO>1 在所有 config 未胜出（延续 v4/v5 结论）。
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
## 接入建议
v6 H6T16C1S2U1 是当前最优。若接入主接口:
## 测试脚本
## v6 最优 vs FlashInfer (同环境实测)
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
> "v6 全面快于 FlashInfer" 的结论**不准确** — 那批数据在 NH=12 group_size=6
### 差距根因 (ncu 微观对比, 16×512)
| 指标 | v6 H6T16C1S2U1 | FlashInfer |
|------|----------------|------------|
| Registers/Thread | **159** | **62** |
| Achieved Occupancy | 25% | **62.5%** |
| Blocks/SM | 3 | **10** |
| SM Throughput | 49% | **61.6%** |
| Stall Long Scoreboard | 1.44 | **0.21** |
| Stall Wait | 1.07 | 0.64 |
| DRAM Throughput | 100% | 100% |
| Duration | 61.7us | 25.6us |

## 测试脚本
- `test/profile/_bench_v6_best.py` — v6 最优 vs v3 对比
- `test/profile/_verify_v6best.py` — v6 最优配置正确性验证
## unroll 全去实验 (用户提问)
**结论: 全去 unroll 严重变差 2.6-3.6x。**

## 来源：`2026-08-12-flash-decoding-v6-4warp.md`

**原题：** flash_decoding v6: 4-warp block (v5 基础上减线程数 + s_q 转 T)

## 动机
## 改动 (vs v5)
| 组件 | v5 | v6 |
|------|----|----|
| BLOCK_DIM | 256 (8 warps) | **128 (4 warps)** |
| NW | 8 | **4** |
| Q load | `if (warp_id < HEADS)` 每 warp 1 head | `for(h=warp_id; h<HEADS; h+=NW)` 循环分 head |
| 归约 | warp0/1 并行分半 | **warp0 串行合并 1,2,3** (4 warp 直接简单) |
| reduce kernel | 256 threads | **独立 256 threads** (parallel 128, reduce 256) |
| s_q 类型 | float | **T (bf16)**, 计算时 copy_4d 转 float4 |
| HEADS 循环 | 不展开 | **`#pragma unroll`** (寄存器压力已降) |

## ncu 对比 (batch=16, totlen=2048)
```
指标                        v4_H6T8C1   v5_H6T8C1   v6_H6T8C1
──────────────────────────────────────────────────────────────
Duration (us)                395.1       545.0       285.9   ← 最优
Block Limit Registers          2           1           3
Block Limit Shared Mem         2           1           3
Achieved Occupancy           31.5%       16.0%       23.6%
Theoretical Occupancy        33.3%       16.7%       25.0%
Waves Per SM                 21.33       42.67       14.22
SMEM/block (KB)              40.32       40.32       26.30
```

### s_q 转 T 的效果
- Q load 变纯 T 拷贝 (更快), 计算时 copy_4d 转 float4
## 性能对比 (nsys, 模板参数精确匹配)
```
b×totlen   v3_H6T8   v6_C1(6,8)  v6_C1(2,16)  v6_best   cfg        v6/v3
────────────────────────────────────────────────────────────────────────
1×  256      10.7       10.3        6.1        6.1   H2T16C1    0.57x
1×  512      10.1        9.8        7.1        7.1   H2T16C1    0.70x
1× 1024      14.0       18.4       13.3       13.3   H2T16C1    0.95x
1× 2048      22.0       27.5       25.4       25.4   H2T16C1    1.15x
1× 4096      39.2       53.4       49.6       49.6   H2T16C1    1.27x
4× 2048      72.4      101.1      101.7      101.1   H6T8C1     1.40x
8× 1024      72.4      101.1      101.7      101.1   H6T8C1     1.40x
16× 512      72.4      101.1      101.7      101.1   H6T8C1     1.40x
32× 256      72.4      101.1      101.7       89.0   H3T16C8    1.23x
```

### 关键结论
1. **v6 H2T16C1 小 batch 显著优于 v3**：1×256 快 43%（6.1 vs 10.7us），
   1×512 快 30%，1×1024 快 5%。
2. **v6 大 batch 仍慢于 v3**：4×2048/8×1024/16×512 慢 40%（101 vs 72us）。
3. **COALESCE 在 v6 中依然无效**：C4 = 48-144us vs C1 = 10-101us，全面退化。
4. **v6 最优 = v3 的 H2T16 小 batch 路径 + 4-warp 架构**，但大 batch 仍需 H6T8。
## vs v3 最优参数选择建议
| 场景 | v3 最优 | v6 最优 |
|------|---------|---------|
| batch=1, totlen≤512 | H2T16 | **H2T16 (v6 快 30-43%)** |
| batch=1, totlen≥1024 | H6T8 | H2T16 (v6 慢 5-27%) |
| batch≥4 | H6T8 | H6T8 (v6 慢 40%) |

## NUM_STAGES 实验 (用户指出三路缓冲占用 smem)
| 指标 | 3-stage | 2-stage |
|------|---------|---------|
| SMEM/block | 26.30KB | **22.21KB** |
| Block Limit Shared Mem | 3 | **4** |
| Duration (16×2048) | 286us | **265us (-7%)** |

## unroll 实验 (用户指出不应全加)
| 指标 | unroll | no unroll |
|------|--------|-----------|
| Duration (16×2048) | 265us | **264us** |
| Registers/Thread | 144 | 144 |
| Block Limit Registers | 3 | 3 |

## 最终 ncu 指标 (2-stage + no compute unroll, 16×2048)
```
Registers/Thread: 144
Achieved Occupancy: 23.5%  (theoretical 25%, 受寄存器 3-block 限制)
Block Limit Registers: 3
Block Limit Shared Mem: 4   ← smem 瓶颈已破
SMEM/block: 22.21 KB
Duration: 264us  (vs v3 ~304us, 快 13%)
Stall 分解: Wait 1.46, Long Scoreboard 1.15, Short Scoreboard 0.88
```

## 后续优化方向
2. **大 batch 慢 40% 根因**: v6 4-warp 每 block ILP 减半 (4 warps vs v3 8 warps)。
## 测试脚本

## 来源：`2026-08-12-flash-decoding-v7-threadlayout.md`

**原题：** flash_decoding v7: FlashInfer 式线程布局尝试

## 动机
## ncu 对比 (16×512)
| 指标 | v6 | v7 | FlashInfer |
|------|-----|-----|-----------|
| Registers/Thread | 159 | **114** | 62 |
| Occupancy | 25% | **31.25%** | 62.5% |
| SM Throughput | 49% | 45.8% | 61.6% |
| Long Scoreboard | 1.44 | 1.29 | 0.21 |
| Short Scoreboard | — | **1.30** | 0.73 |
| Wait | 1.07 | 1.56 | 0.64 |

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
## 关键结论
**FlashInfer 快的核心不是线程数/寄存器, 而是 block-per-seq 单 kernel 架构**:
## 后续方向
## 测试脚本

## 来源：`2026-08-12-flash-decoding-v8-blockperseq.md`

**原题：** flash_decoding v8: block-per-seq×kv_head 单 kernel (第一阶段)

## 设计 (FlashInfer 架构)
## ncu (16×512)
| 指标 | v8 | FlashInfer |
|------|-----|-----------|
| Registers/Thread | **72** | 62 |
| Theoretical Occupancy | **56.25%** | 62.5% |
| Long Scoreboard | **8.67** | 0.21 |
| SM Throughput | 11.6% | 61.6% |
| DRAM | 100% | 100% |

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
## 第二阶段: cp.async pipeline
### ncu 变化 (16×512)
| 指标 | v8 直接读 | v8 + cp.async |
|------|----------|--------------|
| Long Scoreboard | 8.67 | **1.01** |
| SM Throughput | 11.6% | 17.2% |
| Warps Active | 8.35% | 8.40% |

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

## 完整逼近 FlashInfer 的路线
当前 v6 仍是最优自研 (大 batch 快 v3 36%, 小 batch 快 43%)。
## 测试脚本

## 来源：`2026-08-12-flash-decoding-v9-merge.md`

**原题：** flash_decoding v9: v6 架构 + v8 线程布局融合

## 设计
## ncu (16×512, H6T8C1)
| 指标 | v6 | v9 | FlashInfer |
|------|-----|-----|-----------|
| Registers/Thread | 159 | **114** | 62 |
| Occupancy | 25% | **31.25%** | 62.5% |
| SM Throughput | 49% | 45.95% | 61.6% |
| Short Scoreboard | — | 1.30 | 0.73 |

## 性能 (nsys, par+reduce)
```
b×totlen    v6(us)   v9(us)   FI(us)   v9/v6   FI/v9
──────────────────────────────────────────────────────
1×  256     11.5    19.0    10.0   1.65x   0.52x
1× 2048     18.3    23.6    12.3   1.29x   0.52x
1× 4096     32.4    42.7    15.9   1.32x   0.37x
4× 2048     56.1    63.4    25.5   1.13x   0.40x
16× 512     61.4    67.2    25.7   1.09x   0.38x
32× 256     67.0    72.2    25.5   1.08x   0.35x
```

**v9 全面慢于 v6 (8-65%)**。
## 失败原因
## 结论
## 三层架构矛盾总结
| 架构 | 优势 | 瓶颈 |
|------|------|------|
| block-per-KV-block (v6) | grid 大 | reg 159 → occupancy 25% |
| block-per-seq (v8) | reg 72, occ 56% | grid 小 (batch×nkvh) |
| v6+v8 融合 (v9) | 折中 | shuffle + 冗余读 + reg 114 |

**v6 仍是最优自研** (大 batch 快 v3 36%, 小 batch 快 43%)。
## 测试脚本

