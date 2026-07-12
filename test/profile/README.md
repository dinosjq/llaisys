# Inference Operator Profiling

端到端推理 pipeline 的逐算子耗时 profiling 系统。

## 快速开始

```bash
# 编译（带 profiling）
xmake f --nv-gpu=y --profiling=y -c && xmake && xmake install
pip install -e ./python/

# 运行 profile
python test/profile/profile_model.py --model <path> --max_steps 128
```

`--profiling=n` 关闭 profiling（零开销）。

## 怎么工作

1. Qwen2 构造时向 `OpProfiler` 注册完整的算子序列（每层 17 个算子 × 28 层 + pre/post）
2. 每个算子入口自动调用 `PROFILE_STEP()`，推进 profiler 的位置计数器
3. 事件循环用 `PROFILE_BEGIN/END` 包裹 `forward()`
4. 在 decode step {1, 32, 128, 256, 512, 1024} 处自动采样 CUDA event 耗时
5. 模型销毁时（`Qwen2::stop()`）dump `profile_qwen2.json`

## 输出格式

```json
{
  "model": "qwen2",
  "snapshots": [
    {
      "step": 1, "phase": "decode",
      "ops": [
        {"name": "embed", "elapsed_ms": 0.123},
        {"name": "attn_norm", "elapsed_ms": 0.456, "layer": 0},
        {"name": "linear:q_proj", "elapsed_ms": 0.234, "layer": 0}
      ]
    }
  ]
}
```

## CLI

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | (required) | 模型路径 |
| `--max_steps` | 1024 | 每 prompt 最大生成 token 数 |

## Prompt

5 条固定 prompt，覆盖短/中/长输入：

- "What is the capital of France?" (short)
- "Explain how backpropagation works..." (medium)
- "You are designing a GPU inference system..." (long)
- "Write a Python function to compute..." (code)
- "Summarize the transformer architecture..." (medium)
