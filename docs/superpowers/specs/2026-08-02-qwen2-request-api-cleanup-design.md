# Qwen2 旧请求 API 清理设计

## 背景

正确性基础阶段为 Qwen2 引入了独立的 opaque request handle：

`Submit → Await → Abort/Release`

每次 `Submit` 都创建独立 `Sequence`，请求的采样参数、取消状态和生命周期不再按 prompt 共享。KV Cache
复用仍由 `BlockManager` 按完整 cache block 管理。

旧路径仍保留以下语义：

`ModelInfer/ModelAbort → Qwen2::request/abort(token_ids) → _hash_2_seq`

它通过 prompt hash 复用 `Sequence`，与新的请求隔离模型冲突。仓库内生产代码和测试已经全部迁移到
request handle，旧路径没有调用者。

## 决策

完整删除旧请求 API，不保留 ABI 兼容 shim。该变更会主动中断仓库外直接链接
`llaisysQwen2ModelInfer` 或 `llaisysQwen2ModelAbort` 的程序。

不采用以下方案：

- 仅标记 deprecated：会继续保留错误语义、重复状态和死代码。
- 旧符号转接 request handle：旧接口没有显式 handle，无法可靠表达独立请求的 await、abort 和 release。

## 保留的职责边界

### 模型生命周期

保留：

- `llaisysQwen2ModelCreate`
- `llaisysQwen2ModelDestroy`
- `llaisysQwen2ModelWeights`
- C++ 内部 `Qwen2::start()` 和 `Qwen2::stop()`

模型构造时启动 worker，销毁时停止 worker。此次清理不改变 worker 生命周期。

### 请求生命周期

保留：

- `llaisysQwen2RequestSubmit`
- `llaisysQwen2RequestAwait`
- `llaisysQwen2RequestAbort`
- `llaisysQwen2RequestRelease`
- C++ `Qwen2::submit/await/abort(qwen2_request_t)/release`
- `_active_requests` 与 `_request_lock`

每个 request handle 独占一个 `Sequence`。正常完成、异常和取消路径最终都必须执行一次 release。

### KV Cache 生命周期

KV block 的命中、引用计数、共享和淘汰继续由 `BlockManager` 管理。删除 prompt-hash request map
不会关闭 Prefix Cache，也不会复制 KV Cache 管理逻辑。

## 删除范围

### C API

从 `include/llaisys/models/qwen2.h` 和 `src/llaisys/models/qwen2.cc` 删除：

- `llaisysQwen2ModelInfer`
- `llaisysQwen2ModelAbort`

### C++ 模型

从 `src/models/qwen2.hpp` 和 `src/models/qwen2.cpp` 删除：

- `Qwen2::request(...)`
- `Qwen2::abort(int64_t *token_ids, size_t ntoken)`
- `_hash_2_seq`
- `_hash_lock`

新的 request handle 路径不依赖这些成员。

### Python ctypes

从 `python/llaisys/libllaisys/models/qwen2.py` 删除旧 C 符号的 `argtypes/restype` 注册。

同时把 `python/llaisys/models/qwen2.py` 中过时的
`"llaisysQwen2ModelInfer failed"` 错误信息改为
`"llaisysQwen2RequestAwait failed"`。

### 文档

更新当前架构说明和正确性阶段交付报告，使其只描述 request handle 路径。历史计划文档保持不变，
避免重写历史决策记录。

## 数据流

清理后的唯一推理路径：

1. Python `generate()` 或 `generate_async()` 调用 `RequestSubmit`。
2. C API wrapper 持有 `Qwen2` 和 `qwen2_request_t`。
3. `Qwen2::submit()` 创建独立 `Sequence` 并交给 `Scheduler`。
4. `RequestAwait` 等待该 Sequence 生成下一个 token。
5. 正常结束或异常时调用 `RequestRelease`；取消时先调用 `RequestAbort`，阻塞 await 结束后再 release。
6. `BlockManager` 独立判断是否复用已有 KV block。

不存在 prompt hash 到 `Sequence` 的全局映射。

## 错误处理

- 无效 model、request 或提交参数继续通过现有 C API 返回空 handle、`-1` 或 no-op。
- C++ 异常不得越过 C ABI；现有 try/catch 保持不变。
- 删除旧 API 后，不提供运行时迁移警告；编译或动态符号查找失败即表示调用者必须迁移。
- 新 request handle 的异常安全、取消等待和单次释放语义不变。

## 测试设计

采用测试先行：

1. 新增 API 表面测试，要求共享库不再导出 `llaisysQwen2ModelInfer/Abort`，同时仍导出
   `RequestSubmit/Await/Abort/Release`。删除前该测试应因旧符号仍存在而失败。
2. 删除旧实现后重新构建 NVIDIA 共享库，使 API 表面测试通过。
3. 保留并运行已有新路径测试：
   - invalid submit
   - request lifetime 与异步取消
   - Server temperature 与并发取消隔离
   - Core Prefix Cache、Sampling、Chunked Prefill
   - NVIDIA Top-K
   - 端到端 greedy inference
4. 搜索源码，确认旧符号、旧方法和 `_hash_2_seq/_hash_lock` 不再存在于活动代码。

没有旧测试需要迁移或删除：现有 inference、benchmark、profiler 和 Server 测试都通过
`generate()/generate_async()` 间接覆盖新 handle 路径。

## 非目标

- 不修改 Scheduler、BlockManager 或采样算法。
- 不继续后续 workspace、融合或 GPU sampling 优化。
- 不调整 `target.md` 或既有推理优化计划。
- 不为仓库外调用者提供兼容层。

## 回滚

该清理使用独立 commit。回滚该 commit 即可恢复旧 C ABI 和 prompt-hash request 路径，不影响此前已完成的
正确性修复。
