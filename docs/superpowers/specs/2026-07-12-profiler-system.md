# Inference Operator Profiling System — Design Spec

Date: 2026-07-12

## Context

需要量化推理 pipeline 中各算子在不同 KV cache 长度下的耗时分布，定位长上下文推理的性能瓶颈。

## Goals

- 算子入口自动计时（PROFILE_STEP 宏），新模型无需额外改造
- 编译期可选（`--profiling=y`），关闭时零开销
- 在 decode step {1, 32, 128, 256, 512, 1024} 处自动采样
- JSON 输出 per-phase per-step per-operator 统计
- 纯 C++ 内部机制，不暴露给 Python

## Architecture

### OpProfiler 单例

```cpp
class OpProfiler {
    static OpProfiler& instance();

    // 模型注册（构造时调用一次）
    void register_sequence(const std::string& model, size_t layers,
                           const std::vector<std::string>& per_layer_ops,
                           const std::vector<std::string>& pre_ops,
                           const std::vector<std::string>& post_ops);
    void set_sample_points(const std::vector<int>& points);

    // 事件循环中调用
    void begin_forward(bool is_prefill);
    void end_forward(int decode_step);  // prefill 时传 0
    void step();                         // 每个算子入口调用
};
```

### 宏

```cpp
#define PROFILE_STEP()   OpProfiler::instance().step()
#define PROFILE_BEGIN(p) OpProfiler::instance().begin_forward(p)
#define PROFILE_END(d)   OpProfiler::instance().end_forward(d)
```

关闭时展开为 `((void)0)`。

### 模型注册格式（Qwen2 示例）

```cpp
OpProfiler::instance().register_sequence("qwen2", 28,
    { // per_layer_ops (16 ops per layer)
      "attn_norm", "linear:q_proj", "linear:k_proj", "linear:v_proj",
      "rope:q", "rope:k", "kv_cache_move:k", "kv_cache_move:v",
      "paged_attn", "linear:o_proj", "add:attn",
      "mlp_norm", "linear:gate_proj", "linear:up_proj",
      "swiglu", "linear:down_proj", "add:mlp",
    },
    { // pre_ops (before layer loop): "embed" },
    { // post_ops (after layer loop): "out_norm", "linear:lm_head", "top_k" }
);
```

### 算子入口

```cpp
void linear(tensor_t out, ...) {
    PROFILE_STEP();
    // ... 原逻辑
}
```

### 事件循环

```cpp
int decode_step = 0;
while (running) {
    auto [seqs, is_prefill] = scheduler->schedule();
    if (seqs.empty()) continue;

    PROFILE_BEGIN(is_prefill);
    forward(...);
    model->scheduler->postprocess(...);
    PROFILE_END(is_prefill ? 0 : ++decode_step);
}
```

### 采样触发

`end_forward(decode_step)` 内部检查 decode_step 是否匹配 sample_points。匹配时：
1. `cudaDeviceSynchronize()`
2. 计算当前窗口内所有 CUDA event 的 elapsed 时间
3. 快照存入 `_snapshots`
4. 窗口清零，等待下一个采样点

### 输出

模型析构时自动调用 `dump_json("profile_qwen2.json")`，内容：

```json
{
  "model": "qwen2",
  "snapshots": [
    {"step": 1, "phase": "decode", "ops": [
      {"name": "linear:q_proj", "elapsed_ms": 0.123, "layer": 0}
    ]}
  ]
}
```

## Files

| File | Purpose |
|---|---|
| `src/profiler/profiler.hpp` | OpProfiler + ScopedTimer + macros |
| `src/profiler/profiler.cpp` | 实现 |
| `src/ops/*/op.cpp` (12 files) | 添加 PROFILE_STEP() |
| `src/models/qwen2.cpp` | 注册序列 + PROFILE_BEGIN/END |
| `xmake.lua` | `--profiling=y` 选项 + target 集成 |
| `test/profile/profile_model.py` | Python 驱动脚本（正常推理，不感知 profiler） |

## Build

```bash
xmake f --nv-gpu=y --profiling=y -cv && xmake && xmake install
python test/profile/profile_model.py --model <path> --max_steps 128
# 输出: test/profile/results/profile.json (由 C++ 析构自动生成)
```
