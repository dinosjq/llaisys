# 基线与正确性

> 合并自 `2026-08-01-00-baseline.md`、`2026-08-01-01-correctness-foundation.md`。

## 1. 环境与构建

| 项 | 值 |
|---|---|
| 日期 | 2026-08-01 |
| Commit | `7fc100532b1ee6a958127eb65d7984eb5aaed622` |
| 分支 | `opt/inference-system`（当时隔离 worktree） |
| OS | Linux 6.18.33.2-microsoft-standard-WSL2 x86_64 |
| GPU | RTX 4060 Laptop，8188 MiB |
| 驱动 | 592.82 |
| CUDA | 13.1（nvcc 13.1.115） |
| Nsight Systems | 2025.5.2.266 |
| Python | venv 3.12.3；`PYTHONPATH=$PWD/python` |
| 模型 | `/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B` |

```bash
XMAKE_ROOT=y xmake f --nv-gpu=y -cv
XMAKE_ROOT=y xmake && XMAKE_ROOT=y xmake install
```

构建 PASS（既有 `add_cuflags("-Wno-unknown-pragmas")` 告警可忽略）。

## 2. 正确性基线

| 命令 | 结果 |
|---|---|
| `test/test_runtime.py --device nvidia` | PASS |
| `test/test_tensor.py` | PASS |
| `test/ops/{add,embedding,linear,rms_norm,rope,swiglu,paged_attention,topk}.py --device nvidia` | PASS |
| greedy `benchmark_infer.py`（3 prompts，max_steps 16，对齐 HF） | PASS 3/3 |
| `test/test_server.py`（健康检查 / chat / completion / 取消恢复） | PASS |
| `xmake run llaisys-core-test` | PASS |

## 3. 初始化与端到端性能锚点

初始化（`measure_initialization.py`，含 `torch.cuda.synchronize`）：

| 指标 | 值 |
|---|---|
| wall time | **2.173 s** |
| GPU 显存增量 | **4554 MiB** |
| 可用显存 前/后 | 7096 → 2542 MiB |

P0 / D0 / D1（`nsys_profile.py --no-warmup`，每 case 新进程）：

| Case | 输入 | 请求输出 | 实际输出 | Wall | tok/s | Wall/输出 token |
|---|---:|---:|---:|---:|---:|---:|
| P0 | 256 | 1 | 1 | 0.28 s | 3.6 | 280 ms |
| D0 | 1 | 128 | 75（EOS） | 1.81 s | 41.4 | 24.1 ms |
| D1 | 1024 | 128 | 128 | 3.85 s | 33.3 | 30.1 ms |

串行 greedy 三 prompt 聚合：平均延迟 435.4 ms，TTFT 113.7 ms，TPOT 27.24 ms，36.7 tok/s。

## 4. Nsight kernel 基线

| Case | Kernel 启动次数 | Kernel 总耗时 | 主导 kernel |
|---|---:|---:|---|
| P0 | 621 | 125.651 ms | BF16 GEMM 75.682 ms |
| D0 | 44,447 | 1515.339 ms | BF16 GEMV 1183.502 ms |
| D1 | 75,876 | 3323.371 ms | BF16 GEMV 1737.278 ms |

原始报告当时在 worktree `.superpowers/sdd/task-1-artifacts/`（nsys 二进制不入库）。

## 5. 测量限制

- WSL2 笔记本 GPU、单次运行、未锁频；作比较锚点非统计结论。
- D0 在 75 token 后 EOS；无「忽略 EOS」公共 API。
- 同进程串行跑 P0→D0→D1 曾卡死；协议改为每 case 独立进程。

## 6. 正确性基础修复（Task 2）

### 缺陷

- Prefix cache：`check_equal` 误用索引 `1`。
- Sampling：nucleus 内均匀抽样，未保留越过 `top_p` 的边界 token。
- Chunked prefill：队首 prompt 超预算时提前 `break`。
- 旧 prompt-hash API：相同 prompt 共享 `Sequence`。

### 设计

- `sampling.hpp`：有界 top-k、temperature softmax、nucleus、加权采样；每 Sequence 独立 RNG；batch 按最大 K 做一次 device top-k，再按行截断。
- Request handle：`submit/await/abort/release`；独立 Sequence；KV 仅 `BlockManager` 复用。
- 删除旧 `llaisysQwen2ModelInfer/Abort`（无外部兼容层）。

### 验收要点

- 前缀逐元素匹配、block 复用、greedy/temperature/top-p 确定性、多轮 chunked prefill。
- 并发相同 prompt；取消隔离；`temperature=0` 不被替换。
- 取消时 shield 阻塞 `Await`，abort 后再 release C handle。
- NVIDIA top-k：并行 radix；BF16 `(1,151936) K=100` 与 `torch.topk` 一致，中位延迟 **1.2815 ms**（min 1.2780 ms）。
- Submit：先 model submit，再分配 C wrapper；wrapper 失败则 release 已提交请求。

### 后续对比约定

同 GPU/模型/长度/生成设置/`--no-warmup`/新进程；报告初始化与 P0/D0/D1 wall、tok/s、kernel 启动与主导 kernel；不提交 profiler 二进制。
