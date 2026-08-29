# LLAISYS

**LLAISYS** 是一个教育用大模型推理引擎：C++17 后端（导出 C API 的共享库）+ Python ctypes 前端。

从零实现了推理运行时架构、请求级连续批处理、分页 KV Cache、Flash Decoding 与 INT8 量化（W8A8 + KV Cache），支持 Qwen2 / Llama 系列模型，可在 RTX 4060（sm_89）到 NVIDIA B300（sm_103, 268GB）上运行。

## 特性

- **分层架构**：`Model → Context → Layer → Operator`，解耦模型组装、运行时状态、权重管理与算子后端；预分配 Workspace + Tensor View 复用中间显存。
- **连续批处理**：请求级调度（Token/Sequence Budget）、Chunked Prefill、请求抢占。
- **分页 KV Cache**：Block Table + 引用计数 + 块前缀哈希实现跨请求 KV 复用。
- **Flash Decoding**：基于 Paged KV Cache 的 decode attention kernel（cp.async 流水、Online Softmax、分块归并）。
- **INT8 量化**：W8A8（权重 per-channel + 激活 per-token）+ INT8 KV Cache + Walsh-Hadamard 旋转抑制 outlier；`rms_norm_quant`/`swiglu_quant` 融合量化。
- **CUDA 优化**：手写 dp4a INT8 GEMV、Gemv SIMT、向量化访存、多 block 拆行量化、topk k=1 快路径。
- **OpenAI 兼容服务**：FastAPI + SSE 流式 `/v1/chat/completions`。

## 构建

依赖：`xmake`、CUDA Toolkit（NVIDIA 路径）、`torch` / `transformers` / `ml_dtypes`（Python）。

```bash
source venv/bin/activate

# NVIDIA GPU 构建（默认 sm_89；B300 见下方分支说明）
xmake f --nv-gpu=y -c
xmake
xmake install            # 复制 libllaisys.so 到 python/llaisys/libllaisys/
pip install -e ./python/ # 可编辑安装 Python 包

# CPU-only
xmake f -c && xmake && xmake install
```

> 注意：`xmake install` 必须执行，否则 `.so` 不会复制进 Python 包。

## 测试

```bash
python test/test_runtime.py --device nvidia   # 运行时 API
python test/test_tensor.py                    # Tensor 运算
python test/test_ops.py                       # 算子全量测试
python test/ops/argmax.py                     # 单个算子（可 --device nvidia）
python test/test_infer.py --model <path> --device nvidia --test   # 单请求推理
python test/test_infer_batch.py --model <path> --device nvidia     # 批量推理
python test/test_server.py --model <path> --device nvidia          # 服务测试
```

详细命令见 [`docs/常用命令.md`](docs/常用命令.md)，测试工具见 [`test/test_utils.py`](test/test_utils.py)。

## 目录结构

```
src/            C++ 后端（device / core / tensor / ops / kv_cache / scheduler / sequence / models）
include/        C API 头文件（llaisys.h）
python/llaisys/ Python 前端（ctypes 绑定 + Pythonic 封装 + server）
test/           脚本式测试与基准
docs/           设计文档、优化报告、踩坑记录
xmake/          构建配置（cpu.lua / nvidia.lua）
```

## 分支说明

- **main**：通用代码（算子、调度、量化、修复与优化），面向小卡（RTX 4060 / sm_89，`src/config.hpp` 默认值）。
- **b300**：main + B300（Blackwell Ultra / sm_103）专用配置（KV cache 8192 blocks、batch 32768 token / 256 seq）与面向 B300 的优化实验。B300 构建需将 `xmake/nvidia.lua` 的 `-gencode` 改为 `sm_103`，并同步 `src/config.hpp` 调大（见 `docs/optimization/10-b300-porting-and-stress.md`）。

## 文档

- [`CLAUDE.md`](CLAUDE.md) / [`AGENTS.md`](AGENTS.md)：架构总览、算子清单、构建测试约定与常见坑。
- [`docs/optimization/`](docs/optimization/)：优化报告（Flash Decoding、GEMV、W8A8、B300 移植与压测等）。
- [`docs/record/`](docs/record/)：历史踩坑记录。

## License

MIT（见 [LICENSE](LICENSE)）。
