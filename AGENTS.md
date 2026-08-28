# AGENTS.md

LLAISYS 是一个教育用 AI 推理框架：C++17 后端（导出 C API 的共享库）+ Python ctypes 前端（`python/llaisys/`）。

> **架构总览、分层设计、算子清单、模型推理流程**见 [CLAUDE.md](CLAUDE.md)，本文档只补充构建/测试/约定的关键细节与常见坑，不重复架构内容。

## 环境准备

- 所有命令前先激活 venv（否则找不到 `torch` / `llaisys` 包）：
  ```bash
  source ~/llaisys-main/venv/bin/activate
  ```
- 常用命令速查见 [`docs/常用命令.md`](docs/常用命令.md)；本地模型路径固定为 `/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B`。

## 构建

```bash
xmake f --nv-gpu=y -c    # CPU-only: xmake f -c
xmake
xmake install            # 必须！否则 .so 不会复制进 python/llaisys/libllaisys/
pip install -e ./python/ # 可编辑安装 Python 包
```

- NVIDIA 路径由 `--nv-gpu=y` 门控（定义 `ENABLE_NVIDIA_API`，引入 CUDA 工具链）。CPU-only CI 见 `.github/workflows/build.yaml`。
- **WSL 用户注意**：`xmake` 必须在 WSL 原生终端编译（Windows 侧 UNC 路径会报 lock-file 错误）；跑 Python 脚本也必须用 WSL Python。详见 [`memory/wsl-execution.md`](memory/wsl-execution.md)。

## 格式化

用封装脚本，**不要**直接裸调 `clang-format`：

```bash
python scripts/format.py                     # 格式化 git 暂存区文件
python scripts/format.py --path src/         # 格式化指定路径
python scripts/format.py --check --path src/ # 只检查不修改
python scripts/format.py --ref HEAD~1        # 格式化某 commit 以来修改的文件
```

C/C++ 用 `clang-format`（`-style=file`，见根目录 `.clang-format`），Python 用 `black`。

## 测试

- 测试是脚本式（非 pytest），`sys.path` 注入仓库根后 `from test_utils import ...`；绝大多数脚本接受 `--device cpu|nvidia` 与 `--profile`。
- 测试工具（[`test/test_utils.py`](test/test_utils.py)）：
  - `random_tensor(shape, dtype, device, ...)` / `zero_tensor` / `arrange_tensor` → 返回 `(torch.Tensor, llaisys.Tensor)` 配对张量
  - `check_equal(llaisys_result, torch_answer, atol=1e-5, rtol=1e-5)` → 失败打印两侧值；**不支持负 stride**
  - `benchmark(torch_func, llaisys_func, device, warmup, repeat)`
- 端到端推理（正确性以人工对齐 HF 输出为准）：
  ```bash
  python test/test_infer.py --model <path> --device nvidia --test
  python test/test_server.py --model <path> --device nvidia --port 8321
  ```
- 性能基准与 nsys profile 命令见 `docs/常用命令.md`；优化前必须先按 [`memory/perf-baseline-workflow.md`](memory/perf-baseline-workflow.md) 建立 before/after 基线。

## 代码风格

- 头文件一律 `#pragma once`；函数/变量 `snake_case`，类 `PascalCase`；命名空间 `llaisys::ops` / `llaisys::core` / `llaisys::model` 等。
- 算子固定布局：`src/ops/<name>/op.hpp`（声明）→ `op.cpp`（`CHECK_SAME_*`/`ASSERT` 校验 → `context().setDevice` → `switch(deviceType())` 分发）→ `cpu/` 与 `nvidia/`（`.cu` 用 `#ifdef ENABLE_NVIDIA_API` 包裹）。
- 新增算子的完整链路（C++ → C API → ctypes → Python 包装 → 测试）见 `CLAUDE.md`「Adding a new operator」。

## 常见坑（改代码前必读）

1. **构建后忘记 `xmake install`**：`.so` 不会进 Python 包，改 C++ 后 Python 层跑的还是旧库 → 先 `xmake install` 再测。
2. **`test/ops/topk.py` 已损坏**：它 import 不存在的 `benchmark.ops.shape_profiles`，运行会 ImportError——不是你的改动引起的，别去修。
3. **`test_infer.py --test` 不真正断言**：token 相等的断言被注释，`--test` 只做贪心采样 + HF/LLS 双端人工对比。
4. **NVIDIA-only 算子**：`kv_cache_move`、`linear_w8a16`、`quantize_w8`、`dequant_w8` 的 CPU 路径抛 `EXCEPTION_UNSUPPORTED_DEVICE`；`quantize_weights=True` 在非 NVIDIA 上直接 throw。
5. **`paged_attention` decode 分支**（flash decoding）必须传 `attn_acc/attn_sum/attn_max` 三个临时 buffer；prefill 分支不需要。
6. **`docs/format.md` 是误导**：文件名像代码格式文档，实际内容是 profile 输出格式示例，别引用来解释 clang-format。
7. **bf16 权重加载**需要 `ml_dtypes` 已安装，否则显式抛 RuntimeError。
8. **历史回归点**（`docs/record/`）：`max_new_tokens` 曾因未贯穿到 C++ 导致已完成 sequence 不被调度器清理——动序列生命周期代码时注意参数贯穿。
9. **模型层易错点**（见 `memory/` 与过往修复）：`llaisysLinear` 的 `bias==nullptr` 是合法无偏置路径；`Sequence::_cached_ntoken` 必须初始化为 0；`Sequence::last_block_token_num()` 用 `KV_CACHE_BLOCK_SIZE`（不是 `KV_CACHE_BLOCK_NUM`）。
10. **`docs/improve/` 里的 NF4 / Hadamard** 是训练营作业题文档，不是已实现功能；量化现状是 NVIDIA 专用 W8A16（`linear_w8a16` 等）。
