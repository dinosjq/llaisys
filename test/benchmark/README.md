# Inference Performance Benchmark

端到端推理性能对比（HuggingFace vs LLAISYS），支持单请求串行和并发多请求。

## 快速开始

```bash
xmake f --nv-gpu=y -cv && xmake && xmake install
pip install -e ./python/

python test/benchmark/benchmark_infer.py --model <path> --warmup 1 --repeat 1
python test/benchmark/benchmark_batch_infer.py --model <path> --warmup 1 --repeat 1
```

## 目录结构

```
test/benchmark/
├── README.md
├── infer_utils.py               ← 推理 benchmark 共享模块
├── benchmark_infer.py           ← 单请求串行
├── benchmark_batch_infer.py     ← 并发请求
└── results/                     ← JSON 输出（已 gitignore）
```

运行产物也可写到 `test/results/`（nsys / sqlite / 批次结果），同样已 gitignore。

## CLI

| 参数 | 类型 | 默认值 | 说明 |
|---|---|---|---|
| `--device` | `nvidia` | `nvidia` | CUDA 仅用 |
| `--model` | path | — | 模型路径 |
| `--input-lens` | str | `32,128,512` | 精确输入 token 长度列表 |
| `--prompts-per-len` | int | `2` | 每个长度的用例数 |
| `--max_steps` | int | `64` | 最大生成 token 数 |
| `--force-max-tokens` | flag | — | HF `min_new_tokens=max_new_tokens` |
| `--warmup` / `--repeat` | int | `3` / `5` | 对称 warmup；取最佳 wall |
| `--test` | flag | — | greedy + token 比对 |
| `--skip-padded` | flag | — | 仅 batch 脚本：跳过 HF padded 次对照 |
| `--backend` | `both`/`hf`/`llaisys` | `both` | 分后端；8GB 卡建议 `hf` 后再 `llaisys --from-json` |
| `--from-json` | path | — | `--backend llaisys` 时读取 HF 半程结果 |

### 单请求串行

```bash
python test/benchmark/benchmark_infer.py --model <path> \
  --input-lens 32,128,512 --prompts-per-len 2 \
  --max_steps 64 --warmup 3 --repeat 5 [--force-max-tokens]
```

- 输入按 **token 长度实时构造**，HF/LLAISYS **共用同一 `input_ids`**
- 主指标：wall、out tokens、tok/s；speedup = LLAISYS ÷ HF serial
- `--test`：greedy 下比对完整 token 序列

### 并发多请求

```bash
python test/benchmark/benchmark_batch_infer.py --model <path> \
  --input-lens 32,128,512 --prompts-per-len 2 \
  --max_steps 64 --warmup 3 --repeat 5 [--force-max-tokens]
```

- **主对照**：HF serial ↔ LLAISYS concurrent（continuous batching）
- **次对照**：HF padded batch（可能慢于 serial；不当主 KPI）

### JSON 输出

`results/run_YYYYmmdd_HHMMSS/results.json`：

```json
{
  "environment": {"gpu": "...", "cuda": "..."},
  "config": {"mode": "single-serial", "input_lens": [32,128,512], "force_max_tokens": true},
  "results": {"hf": {...}, "llaisys": {...}, "rows": [...]},
  "speedup_vs_hf_serial": 1.42
}
```
