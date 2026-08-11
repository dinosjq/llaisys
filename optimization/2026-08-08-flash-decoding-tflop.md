# flash-decoding 算子性能分析

日期: 2026-08-09（更新到 warp 并行版）
测试对象: `src/ops/paged_attention/nvidia/flash_decoding_nvidia.cu`
对比基线: FlashInfer 0.6.16 `BatchDecodeWithPagedKVCacheWrapper`（同卡、同模型配置、同输入）
硬件: RTX 4060 Laptop（24 SM, 256 GB/s 显存带宽, Ada Lovelace CC 8.9）

## 测试配置

模型 DeepSeek-R1-Distill-Qwen-1.5B（`config.json`）: nh=12, nkvh=2, hd=128。每个 KV 缓存块存 64 个 token（`TOKEN_NUM=64`）。

FLOPs 公式: `batch × nh × 4 × totlen × hd`。`4` 来自 QK^T（2）+ P@V（2），softmax 不计入。

算子输入：
- Q: `(batch, nh, hd)` — decode 阶段每序列 1 个 token
- K/V cache: `(num_blocks, 64, nkvh, hd)` — 页式存储，通过 block_ids 索引
- block_ids: `(batch, max_block_num)` — 每序列的页表（V3 前缀和格式，首列存累计块数）
- cut_idx: `(batch+1,)` — 每条序列在 Q 里的起始位置
- tot_len: `(batch,)` — 每条序列已生成的 KV 总长度

## 代码结构

`flash_decoding_launch()` 每次调用启动两次 kernel：

```cuda
// Kernel 1: 分块并行计算（按 nkvh 派发）
parallel_kernel<<<(max_block_num, nkvh, batch), 256, smem_0>>>(...)

// Kernel 2: 跨块合并（按 nh 派发）
reduce_kernel <<<(1, nh, batch), 256, smem_1>>>(...)
```

### parallel kernel

**网格按 nkvh=2 派发**（grid.y），每个 block 负责 1 个 KV head × 1 个 KV 块（64 token），但 block 内服务 `rep_max = nh/nkvh = 6` 个 query head。256 线程 = 8 warp：

| warp 0–5 (consumer) | warp 6 (producer K) | warp 7 (producer V) |
|---|---|---|
| 各处理 1 个 query head（`nh = (nh/nkvh)*nkvh + warp_id`） | 从 global 预取下一 token 的 K 到 smem | 从 global 预取下一 token 的 V 到 smem |
| Q 常驻寄存器 `reg_q[4]`，逐 token 做点积 + online softmax + 累加 | 双缓冲 `now ^= 1` | 双缓冲 `now ^= 1` |

每个 token 的循环内：
1. 点积：每个 lane 持有 4 个连续元素（`reg_q[i] * s_k[lane*4+i]`），`warp_reduce` 归约 32 lane → 128 维全点积
2. online softmax：`lane_id==0` 单算 `expf`，`__shfl_sync` 广播给全 warp（不再 32 lane 重复算）
3. 累加器：`reg_acc[4] = alpha*reg_acc[4] + beta*s_v[...]`

一个 KV 块处理完后，每个 consumer warp 把结果写到 global 的三个 workspace buffer（按 `(block, head)` 索引）：
- `attn_acc`: `(total_blocks, nh, hd)` — 各块的注意力累加值
- `attn_sum`: `(total_blocks, nh, 1)` — softmax 分母
- `attn_max`: `(total_blocks, nh, 1)` — softmax 最大值

### reduce kernel

每个 block 按 nh 派发（grid.y = nh = 12），每个 block 合并一条序列一个 head 的所有块结果。256 线程 8 warp 双缓冲：warp 4–7 预取下一块的 acc/sum/max，warp 0–3 做 online softmax merge。最终 warp 0 写出 `attn_val: (batch, nh, hd)`。

### 与旧实现的关键差异

这次重写把之前分析里最大的两个问题都改掉了：

| | 旧实现（按 nh 派发） | 新实现（按 nkvh 派发） |
|---|---|---|
| 网格 | grid.y = nh = 12，每 block 1 个 head | grid.y = nkvh = 2，每 block 6 个 head（warp 并行） |
| K/V 读取 | 同份 KV 被 6 个 head-block 各读一次（6× LDG） | 一份 KV 进 smem 服务 6 个 head（1× LDG） |
| Q 存放 | 放 smem `s_q`（6×d 空间） | 放寄存器 `reg_q[4]`（省 smem） |
| softmax | 32 lane 各算一遍 expf | lane 0 单算 + shuffle 广播 |
| 双缓冲 | 4 槽 `now ^= 4` | 2 槽 `now ^= 1`（每 token 一槽） |

## 性能对比

### 延迟和 TFLOPS

所有测试 bf16，warmup 后取最速。单位 us。

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

**batch=1 小 totlen 时基本打平。** 1×256: 49.1 vs 48.3。FlashInfer 的 batch_decode 有框架层固定开销（workspace、indptr/indices 解释），你的算子没有，但你的并行度也低——这部分互相抵消了。

**batch=1 大 totlen 仍有差距。** 1×2048: 75.8 vs 53.9。FlashInfer 的 batch=1 延迟几乎不随 totlen 增长（48.3→53.9us 从 256 到 2048），它单 kernel + cp.async 流水把 KV 读取延迟完全藏住。你从 49us 涨到 75us，随块数线性增长。

**batch 越大差距越大。** 16×2048: 804.9 vs 144.3，差 5.6 倍（旧版是 8.8 倍，gap 收窄了）。你的耗时仍近线性随 batch 增长，FlashInfer 是亚线性（batch=4→8→16，延迟 +32%→+60% 而非 +100%）。

**TFLOPS 天花板。** LLAISYS 在 0.1–0.3，FlashInfer 到 1.40（16×2048 时带宽利用率 91%，233 GB/s）。decode 注意力是 memory-bound，0.3 TFLOPS 不是算得慢，是数据供不上。FlashInfer 供得上的原因是单 kernel + 高占用流水。

### 为什么 batch=1 反而退化（相对旧版）

旧版按 nh=12 派发，1×256 有 4 block × 12 = 48 个 block；新版按 nkvh=2 派发，只有 4 × 2 = 8 个 block。8 block × 256 threads = 2048 线程，4060 的 24 SM 大半空闲——**K/V 读取 6×→1× 省下的 L1/LSU 吞吐，被低 occupancy 的并行度不足盖过**。大 batch 时 block 数足够（16×2048 = 32×2×16 = 1024 block），K/V 复用的收益才体现出来。这是"按 nkvh 派发"的自然代价：block 数除以 6，小规模并行度下降。FlashInfer 用 PDL（programmatic dependent launch）和更高占用流水来缓解，我们还没做。

## 与 FlashInfer 的剩余差距

重写后两个实现的对齐情况：

| | LLAISYS（当前） | FlashInfer |
|---|---|---|
| 网格派发 | 按 nkvh=2 ✓ 已对齐 | 按 nkvh |
| K/V 读取 | 1× ✓ 已对齐 | 1×（cp.async 异步加载） |
| softmax | lane0 单算 + shuffle ✓ 已对齐 | lane0 + exp2 |
| 收尾 | 独立 reduce kernel（一次额外 launch + workspace 读写） | 单 kernel 内完成（partition-kv + merge） |
| K/V 加载 | `copy_4d`（同步 LDG → smem） | `cp.async`（异步拷贝，与计算重叠） |
| 流水深度 | 2 槽双缓冲，每 token 同步一次 | 多级 smem tile 流水，整块预取 |
| 同步粒度 | 每 token 一次 `__syncthreads()` | 每 tile 一次 `block.sync()` |

**batch=16, totlen=2048 的具体分解：**

FlashInfer 144.3us 完成 16 条序列 decode，带宽利用率 91%。按 233 GB/s 倒推约传输 33.6 MB，K/V 总量 16×2048×2×128×2 = 16.8 MB，加上 Q/输出/中间数据合理。

你的算子 804.9us。即使全部优化到位，理论上限仍受制于：

1. **每 token 一次 `__syncthreads()`。** hd=128 时每 token 计算只有 4 次 FMA（每 lane），却要等一个 block 同步。FlashInfer 一次处理一个 tile（多 token）再同步，把同步摊薄。这是当前最大的结构差距。

2. **2 槽浅流水。** 预取和计算的重叠窗口只有 1 个 token。FlashInfer 是 num_stages_smem 级流水，整块 KV 在计算前就异步到位。

3. **独立 reduce kernel。** 多一次 launch + workspace 读写的固定开销。batch=1 时尤其明显。

4. **无 PDL。** FlashInfer 用 `griddepcontrol.launch_dependents` 让 reduce 与 parallel 在流上直接接力，省 launch 间隙。

## ncu profiling 结果（旧实现）

> 以下数据来自按 nh=12 派发的旧 kernel（batch=4, totlen=2048）。新 kernel 的 profiling 待重跑，但 stall 原因和 bank conflict 的根因分析仍适用于新实现（`copy_4d` 写 smem 的模式没变）。

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

每发一条指令平均停了多少 cycle：

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

读不冲突：点积各 lane 读不同位置，32 lane 各占不同 bank。
写冲突：`copy_4d` 的 float4 写入，`col = lane_id<<2` 导致每 4 个 lane 写入目标落在同一 bank。新实现保留了 `copy_4d` 这个加载路径，冲突依旧存在。

## 优化方向（按收益）

已完成的：**GQA 按 nkvh 派发**（K/V 读取 6×→1×）、**softmax lane0 单算**（expf 64→2 次）、**Q 移入寄存器**（省 smem、减 bank conflict 面）。

剩余的：

1. **减少同步频率（最大头）。** 每 token 一次 `__syncthreads()`。改为一次处理多个 token（比如 8 个）再同步，或用 `cp.async` + 多级流水，让 K/V 加载与计算重叠。这是拉开和 FlashInfer 差距的关键。

2. **双缓冲加深到多槽流水。** 当前 2 槽、重叠窗口 1 token。加大 `s_k/s_v` 的 stage 数，预取提前数个 token。

3. **batch=1 的并行度问题。** 按 nkvh 派发后 1×256 只有 8 个 block。可以考虑 batch=1 时回退到按 nh 派发（K/V 有 L2 缓存，冗余读代价在单序列下可接受），或增加每 block 的独立工作量。

4. **收尾内联化。** batch=1 时 parallel 最后一块直接写 attn_val，跳过 reduce kernel。

5. **smem 布局 padding。** 给 smem K/V buffer 每行加 padding 错开 bank，消除 float4 写入冲突。

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
