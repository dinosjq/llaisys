# Model / Layer 框架设计规格（修订）

日期：2026-08-16  
状态：对齐实现中  
取代：2026-08-15 规格中「共存策略 A / 硬编码保留」等迁移期条款

## 1. 目标

- `Model` 基类持有全部共用运行时：scheduler / KV / worker、`start`/`stop`、`submit`/`await`/`abort`/`release`、`prepare_*`、`ModelContext`
- 子类（`Qwen2` / 未来 `Llama`）只做：构造与权重、workspace 分配、`forward` 里显式组装语义 Layer
- **删除**硬编码 `forward_legacy` 与 `LLAISYS_QWEN2_LAYER_FORWARD`
- Layer = Embedding / Attn / FFN；禁止按算子拆 Layer
- 编码风格对齐现有 `qwen2`：中文步骤注释、`this->`、成员 `_` 前缀

## 2. 目录

```text
src/
  layers/
    layer.hpp                         # Layer 基类；include models/core/model_context.hpp
    qwen2/qwen2.hpp + qwen2.cpp       # Emb/Attn/FFN + run_qwen2_layer_stack
    llama/llama.hpp + llama.cpp       # 同上（无 bias）
  models/
    core/
      model_context.hpp               # Meta / BatchRuntime / Workspace / BatchPack / WeightRole
      model.hpp + model.cpp           # Model 基类（运行时 + set_weight + prepare_*）
    qwen2/qwen2.hpp + qwen2.cpp       # class Qwen2 : public framework::Model
  sampling/
    sampling.hpp + sampling.cpp
```

删除：`qwen2_pack.hpp`、`weight_role.hpp`、分文件的 emb/attn/ffn/stack、硬编码路径。

## 3. Model 基类（必须上提）

| 能力 | 说明 |
|---|---|
| `_scheduler` / `_k_cache` / `_v_cache` | 调度与 KV 容器 |
| `_request_lock` / `_active_requests` | request 表 |
| `_running` / `_worker` | 事件循环 |
| `start` / `stop` | worker；`start` 内调虚函数 `forward` |
| `submit` / `await` / `abort` / `release` | 生命周期（`ModelRequest`） |
| `prepare_block_table` / `prepare_prefill` / `prepare_decode` | 返回 `BatchPack` |
| `set_weight` | WeightRole → Context 分层槽 |
| `_ctx` | ModelContext |
| `forward(...)` | **纯虚**：子类组装 Layer |

## 4. BatchPack

原 `Qwen2Pack` 改名为 `framework::BatchPack`，放在 `model_context.hpp`（与 `BatchRuntime` 同文件）。字段不变。

## 5. Qwen2 子类职责

- Meta / legacy Weights / workspace 张量袋
- 构造时分配 workspace + KV 视图、填 `_ctx.meta`
- `forward`：`sync_weights` → bind runtime → `run_qwen2_layer_stack`（或 llama）→ sample
- 可选：`LLAISYS_LAYER_STACK=llama` 选 Llama 层组装（纵向切片保留）

## 6. 非目标（本轮仍不做）

- 通用 C ABI 替换 `llaisysQwen2*`
- Llama-3 `rope_scaling`
- 删除 Scheme A legacy Weights 结构（Python 仍可填旧 struct 再 sync）
