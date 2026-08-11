# GEMV 算子实验：decode linear 优化尝试

日期: 2026-08-10
测试对象: `src/ops/linear/nvidia/gemv_nvidia.cu`
目标: 应对 decode 阶段 linear（GEMV）耗时高
硬件: RTX 4060 Laptop（24 SM, 256 GB/s, Ada Lovelace CC 8.9）

## 背景

decode 阶段 `linear: out[m,n] = in[m,k] @ weight[n,k]^T`，m=batch（16-64），
n/k 大（1536-8960）。每 token 28 层 × ~5 linear = 140 次调用，耗时累积。

## GEMV 设计（参照 v3 flash-decoding 优化）

- **模板化**: `<T, K_TILE=256, NUM_STAGES=3>`
- **cp.async**: weight 行用 cp.async 流式加载到 smem（3-stage 深流水）
- **分块**: grid=(ceil(n/8), 1)，每 block 8 个 out 行 × 全部 m
- **全体参与**: 8 warp 各负责 1 个 out 行，无 producer/consumer 分离
- **归约**: 每 warp 对全部 m 做 k 维 dot，warp_reduce
- weight 常驻寄存器（每 lane 8 元素，16 个 mm 复用）

## 优化迭代

| 版本 | 16×1536×1536 | 说明 |
|------|-------------|------|
| 初版（cp.async weight） | 153us | 正确性 ULP 级 PASS |
| + weight reg 复用 | 137us | ncu: short_scoreboard（weight smem）降 |
| + in 进 smem | 161us | **变差**：cp.async in 开销 + smem 增降并发 |
| 最终标量版 | 145us | 回退 in，保留 weight reg |

## 性能对比（最终标量版 vs LLAISYS cublasGemmEx）

```
   m      n      k     gemv   linear(speedup)
  16   1536   1536    145us    35us   0.24x
  16   8960   1536    730us    70us   0.10x
  32   8960   1536   1367us    91us   0.07x
```

## 结论

**标量 GEMV 无法超越 cublas（4-15× 慢）**：

1. **根本瓶颈是计算密度**：标量 FFMA + warp_reduce 每输出 k 次 FMA +
   每 tile 每 mm 一次 5-shuffle reduce；cublas 用 tensor core mma，
   一次算 16×8×16，指令数少一个量级。ncu 确认瓶颈是计算（非访存），
   所有 load 优化（cp.async/smem/reg）都无法弥补。

2. **cublas 已近带宽极限**：decode GEMV 是 memory-bound（读 weight[n,k]）。
   LLAISYS cublasGemmEx 30-92us ≈ 1.8× 理论带宽（4.7MB / 256GB/s ≈ 18us）。
   单次 linear 并不慢。

3. **decode 耗时高是累积**：140 次调用 × 平均 ~50us = ~7ms/step，这是
   decode 的本质（每层读权重），非单次 linear 低效。

## tensor core mma GEMV（第二次尝试）

针对标量瓶颈（计算密度），用 `mma.m16n8k16`（bf16 tensor core）重写：

- grid=(ceil(n/64), ceil(m/16))，每 block 64 个 out 行 × 16 m，8 warp 各 8 行
- A=in[16,K]、B=weight^T[K,8] 经 cp.async 加载到 smem，K 维沿 mma 展开
- **正确性完美（diff=0.00000）**，比标量快 4×（145→35us）

但**仍比 cublas 慢 8-17%**：

```
   m      n      k   gemv_mma  cublas
  16   1536   1536     35.5    30.4
  16   3456   1536     42.7    37.7
  16   8960   1536     73.9    68.6
  16   1536   8960     70.0    66.4
```

cublas 的 mma 内核 + 深流水 + 启发式仍更优。手写 mma 难超越高度优化的 cublas。

## 最终结论

1. **标量 GEMV**：正确（bf16 ULP 级），慢 cublas 4-15×（计算密度不足）
2. **tensor core mma GEMV**：正确（diff=0），快标量 4×，但仍慢 cublas 8-17%
3. **cublas 已是最优**：decode GEMV 单次近带宽极限（30-92us ≈ 1.8× 理论带宽）
4. **decode 耗时高是累积**：140 次调用 × ~50us ≈ 7ms/step，非单次 low-level 问题

## 真正的提升方向

- **算子融合**（mlp gate/up 融合，减少调用次数）
- **权重驻留**（层权重 L2 常驻优化）
- 或接受 cublas（已近带宽极限）

## 环境备注

测试中发现 `xmake f -cv` 会清掉 `--nv-gpu=y` 配置（NVIDIA 未编译，
getDeviceCount=0）。恢复：`xmake f --nv-gpu=y -cv`。已恢复并全量回归通过。
