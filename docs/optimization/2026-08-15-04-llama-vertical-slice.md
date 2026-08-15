# Llama-3.2 纵向切片 — 交付报告

**日期:** 2026-08-16  
**分支:** `feat/model-layer-framework`  
**模型:** `/home/songjq/models/Llama-3.2-3B-Instruct`（经 ModelScope `LLM-Research/Llama-3.2-3B-Instruct` 安装；HF/镜像不可达）

## 目标

同一 Model/Layer 框架接入第三模型；**不改** `Scheduler` / `Sequence` / `BlockManager`。

## 做法

| 面 | 落点 |
|---|---|
| Layer | `src/models/layers/llama/{embedding,attention,ffn,stack}.*`（无 bias） |
| 选择 | 环境变量 `LLAISYS_LAYER_STACK=llama` → `run_llama_layer_stack` |
| 运行时 | 复用现有 Qwen2 C API / worker（共享 decoder runtime） |
| 映射 | `python/llaisys/models/llama_weight_map.py` + `llama.py`（tied embed） |

## 已知限制

- **未实现** Llama-3 `rope_scaling`（`llama3`）；当前仅用 `rope_theta=500000` 标准 RoPE。短上下文冒烟可用，长上下文质量可能偏差。
- 权重安装路径：优先 ModelScope（本机 HF.co / hf-mirror 超时）。

## 验证

```bash
PYTHONPATH=./python:$PYTHONPATH python test/test_llama_smoke.py \
  --device nvidia --model /home/songjq/models/Llama-3.2-3B-Instruct --max-steps 4
```

结果：`generated_tokens=4`，解码 `Hello! How can` → **PASS**（RTX 4060 8GB）。

## 回滚 / 切换

```bash
unset LLAISYS_LAYER_STACK          # 默认 Qwen2 stack
# Llama Python 构造器会设置 LLAISYS_LAYER_STACK=llama
```

## 状态更新（2026-08-16）

- `LLAISYS_LAYER_STACK` **已删除**。
- C++：`src/models/llama/`，`class Llama : public Qwen2`，`override forward` 组装 Llama Layer。
- C/Python：`llaisysModelCreate(LLAMA, …)` + `SetWeight`；不再借用 Qwen2 Weights ABI。
- 终态说明：`docs/optimization/2026-08-16-05-unified-model-api.md`。
