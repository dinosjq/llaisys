# 统一 Model C API + Llama 继承 + Model 内层组装

日期：2026-08-16  
状态：已实施（2026-08-16） · 计划：`docs/superpowers/plans/2026-08-16-unified-model-api.md`  
前置：`docs/superpowers/specs/2026-08-15-model-layer-framework-design.md` 及后续 alignment（Model 基类、Layer 语义块、WeightRole）已落地

## 1. 目标与非目标

### 目标

1. **统一对外 C/Python API**：以 `Model` / `WeightRole` / Request 生命周期为准，去掉 Qwen2 专名 ABI。
2. **`Llama` 成为独立模型类**：`src/models/llama/`，`class Llama : public Qwen2`，主要 `override forward`。
3. **层组装在 Model 内完成**：删除 `run_qwen2_layer_stack` / `run_llama_layer_stack`；`layers/` 只保留语义 Layer 类。
4. **权重直达 Context**：`SetWeight(role, layer, tensor)` → `Model::set_weight`；不再对外暴露 Weights 大结构体，去掉 Scheme A legacy sync 作为上传主路径。
5. **去掉** `LLAISYS_LAYER_STACK` 环境变量切栈。

### 非目标

- 不改 Scheduler / Sequence / BlockManager。
- 不实现 Llama-3 `rope_scaling`。
- 不保留旧 `llaisysQwen2*` ABI 兼容层（教育仓库，本轮直接替换）。
- 不引入配置驱动 graph / 算子级 Layer。

## 2. 总体结构

```text
C/Python
  LlaisysModelArch + LlaisysModelMeta
  llaisysModelCreate / SetWeight / Request*
        │
        ▼
framework::Model                    // scheduler / KV / worker / prepare
  └── Qwen2                         // workspace、采样、Qwen2 层组装
        └── Llama                   // override forward → Llama 层组装

layers/
  qwen2/{Embedding,Attention,FFN}   // 无 run_*_stack
  llama/{Embedding,Attention,FFN}   // 无 run_*_stack
```

实现命名空间为 `llaisys::core`（目录 `src/models/core/`），不再使用 `framework`。

## 3. C API（`include/llaisys/models/model.h`）

### 3.1 类型

- `LlaisysModelArch`：`LLAISYS_MODEL_QWEN2` | `LLAISYS_MODEL_LLAMA`
- `LlaisysModelMeta`：与 `framework::ModelMeta` 字段一致  
  (`dtype, nlayer, hs, nh, nkvh, dh, di, maxseq, voc, epsilon, theta, end_token`)
- `LlaisysWeightRole`：与 `framework::WeightRole` **同序同值**
- 不透明：`struct LlaisysModel` / `struct LlaisysRequest`

### 3.2 函数

| API | 行为 |
|---|---|
| `llaisysModelCreate(arch, meta, device, device_ids, ndevice)` | `QWEN2` → `shared_ptr<Qwen2>`；`LLAMA` → `shared_ptr<Llama>`；句柄持有 `shared_ptr<framework::Model>`（或可动态转换的基类指针） |
| `llaisysModelDestroy` | `stop()` + 释放 |
| `llaisysModelSetWeight(model, role, layer, tensor)` | 调 `Model::set_weight(ctx, role, layer, tensor)`；全局权重 `layer=0` |
| `llaisysModelRequestSubmit` / `Await` / `Abort` / `Release` | 语义同现 Qwen2 Request API（`-2` 取消，`-1` 错误） |

### 3.3 删除

- `include/llaisys/models/qwen2.h`
- `src/llaisys/models/qwen2.cc`（改为 `model.cc`）
- 对外 `LlaisysQwen2Weights` / `llaisysQwen2ModelWeights`
- 一切 `llaisysQwen2*` 符号

## 4. C++ 模型

### 4.1 Qwen2

- 仍负责：构造 meta/workspace/KV/scheduler、`bind_context_runtime`、采样、`forward` 内 **显式**组装：
  - `Qwen2Embedding` → 逐层 `Qwen2Attention` / `Qwen2FFN` → last-token embedding → out RMSNorm → lm_head → sample
- 删除：`_use_llama_stack`、对 `run_*_layer_stack` 的调用、依赖 env 的分支
- `sync_weights()`：不再从 legacy Weights 数组拷贝；若内部仍暂留 `_weights` 仅作过渡，**上传主路径以 SetWeight 为准**。目标态可去掉 `_weights` 与 `init_weight_arrays`（实现计划中单列任务，可与 C API 同轮完成）
- `forward` / 需要被子类替换的辅助方法标为 `virtual`（至少 `forward` 已在 `Model`；`bind_context_runtime` / `sample_from_logits` 按需 `protected` + `virtual`，默认不强制 Llama override）

### 4.2 Llama（`src/models/llama/`）

```cpp
class Llama : public Qwen2 {
public:
    using Qwen2::Qwen2;
    std::vector<int64_t> forward(BatchPack &, std::vector<int64_t> &, bool) override;
};
```

- `forward`：同结构组装，改用 `LlamaEmbedding` / `LlamaAttention` / `LlamaFFN`
- 构造可复用 Qwen2（共享 decoder 容量与 workspace）；差异集中在层实现（无 attn bias 等）
- 不改 Scheduler / Sequence / BlockManager

### 4.3 Layer 目录

- 保留各 `*Embedding` / `*Attention` / `*FFN` 的 `forward(ModelContext &)`
- **删除** `run_qwen2_layer_stack` / `run_llama_layer_stack` 声明与实现
- 诊断 capture（layer0）若仍需要：逻辑移入 `Qwen2::forward`（或测试专用钩子），不放在已删除的 stack 函数里

## 5. Python

- 新绑定：`libllaisys/models/model.py`（Meta / Arch / WeightRole / Create·SetWeight·Request*）
- `models/qwen2.py` / `models/llama.py`：`Create(arch, …)` + HF→role → `SetWeight`；去掉 env 切栈与 `LlaisysQwen2Weights`
- `*_weight_map.py`：保留映射；`assign_*` 改为调用 `SetWeight`（或返回 role 由模型类 Set）
- Llama：缺省不传 attn bias；`tie_word_embeddings` 时对 `OutEmbed` 再 Set 一次 in_embed

## 6. 构建与测试

- `xmake`：加入 `src/models/llama/*.cpp`；C 桥改为 `src/llaisys/models/model.cc`
- 更新所有引用 `qwen2.h` / `llaisysQwen2*` 的测试与脚本
- Layer CPU 单测：直接构造 Layer 或经 Model.forward，不依赖 stack 函数
- 门禁：编译通过；Qwen2 推理冒烟；Llama smoke（现有脚本改新 API）

## 7. 实现顺序（概要）

1. C 头 + `model.cc` + Python ctypes；迁移 Qwen2 Python 到 SetWeight  
2. Qwen2::forward 内联组装；删 stack 函数与 env  
3. 新增 `models/llama`；Create(LLAMA)；Python Llama 改新 API  
4. 删除旧 qwen2 C API / Weights 暴露；修测试与 xmake  
5. 冒烟验证

## 8. 已确认决议

| 项 | 决议 |
|---|---|
| API 形态 | 统一 Model C API（方案 A） |
| 权重上传 | 一步到位 SetWeight(role)，不保留对外 Weights |
| 旧 ABI | 本轮删除，无兼容包装 |
| Llama | `models/llama`，继承 Qwen2，override forward |
| 层组装 | 在 Model::forward 显式组装；删除 layers 下 run stack |

## 9. 开放实现细节（计划阶段可定）

- 句柄内部类型：`shared_ptr<Model>` vs `shared_ptr<Qwen2>`（Llama 向上转换）——推荐持 `shared_ptr<Model>`，Submit 经基类虚表
- Qwen2 内部 `_weights` 是否同轮清掉：推荐同轮清掉，避免双路径
- layer0 capture：仅测试需要时保留在 Qwen2::forward 的可选开关（沿用 `enable_layer0_capture`）
