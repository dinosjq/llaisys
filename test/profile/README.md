# Inference Operator Profiling

端到端推理 pipeline 的逐算子耗时 profiling 系统。

## 快速开始

```bash
xmake f --nv-gpu=y --profiling=y -c && xmake && xmake install
pip install -e ./python/

# 单条 prompt
python test/profile/profile_model.py --model <path> --prompt-index 0

# 全部 prompt
python test/profile/profile_model.py --model <path>
```

## CLI

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--model` | (required) | 模型路径 |
| `--max_steps` | 1024 | 每 prompt 最大生成 token 数 |
| `--prompt-index` | -1 (全部) | 指定单条 prompt (0-4) |

## 采样点

在 decode step {0(prefill), 1, 32, 128, 256, 512, 1024} 处自动快照。

## 输出格式

终端输出（聚合到 operator 级别，28 层求和）：

```
  period: prefill   step: 0

    operator          avg_ms    count    total_ms
    ─────────────────────────────────────────────
    embed               0.16        1       0.16
    attn_norm           2.82       28      79.06
    linear:k_proj       5.82       28     163.06
    top_k               4.54        1       4.54
    ...
    ─────────────────────────────────────────────
    total                                    295.93

  period: decode   step: 1

    operator          avg_ms    count    total_ms
    ─────────────────────────────────────────────
    attn_norm           0.08       28       2.29
    ...
    ─────────────────────────────────────────────
    total                                     52.06
```

- `count` = 调用次数（28 = 28 层求和，1 = 单次调用）
- `avg_ms` = total_ms / count
- `total_ms` = 该 operator 在当前 step 的总耗时

JSON 同时写入 `profile_qwen2.json`。

## Prompt

5 条候选，覆盖短/中/长：

| index | 类型 | 内容 |
|---|---|---|
| 0 | short | What is the capital of France? |
| 1 | medium | Explain how backpropagation works... |
| 2 | long | You are designing a GPU inference system... |
| 3 | code | Write a Python function to compute Fibonacci... |
| 4 | medium | Summarize the transformer architecture... |
