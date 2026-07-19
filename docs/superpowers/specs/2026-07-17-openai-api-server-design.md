# OpenAI-Compatible API Server — Design Spec

> 日期: 2026-07-17 | 状态: 待用户审核

---

## 1. 目标

为 LLAISYS 实现 OpenAI 兼容的推理 API Server，支持：
- `/v1/chat/completions` — 对话补全（SSE streaming + 非流式）
- `/v1/completions` — 文本补全（legacy 兼容）
- `/v1/models` — 模型列表
- Request cancellation（客户端断开时中止推理）
- Logprobs 返回
- Per-request 采样参数（temperature/top_p/top_k）

---

## 2. 架构总览

```
┌─────────────────────────────────────────────────────────┐
│                   Python Layer                          │
│                                                         │
│  FastAPI (async)                                        │
│  ┌────────────┐    ┌──────────────┐    ┌─────────────┐ │
│  │ /v1/models │    │ /v1/chat/    │    │ Request     │ │
│  │            │    │ completions  │    │ Manager     │ │
│  │ GET 列表   │    │ POST 推理    │    │ req_id → c  │ │
│  └────────────┘    └──────┬───────┘    └──────┬──────┘ │
│                           │                   │        │
│                           ▼                   │        │
│                    Qwen2.generate_async()     │        │
│                    async generator            │        │
│                    ┌──────────────────┐       │        │
│                    │ asyncio.Queue    │◄──────┘        │
│                    │ (SSE 桥接)       │  取消信号       │
│                    └────┬─────────────┘                │
│                         │                              │
│                    asyncio.to_thread()                 │
│                         │                              │
├─────────────────────────┼──────────────────────────────┤
│                   C API │ Layer                        │
│                         ▼                              │
│  llaisysQwen2ModelInfer()  llaisysQwen2ModelAbort()    │
│  llaisysQwen2ModelGetLogprobs()                        │
│                         │                              │
├─────────────────────────┼──────────────────────────────┤
│                   C++ Backend                          │
│                                                         │
│  Qwen2::request()                                       │
│    condition_variable.wait() ◄── 真休眠                 │
│                                                         │
│  Worker Thread (event loop)                              │
│    schedule → forward → postprocess                      │
│                  │                                       │
│                  ▼                                       │
│    seq->notify_all()  ◄──── 唤醒等待线程                 │
│                                                         │
│  Scheduler::abort() ◄──── 取消序列 + 释放 KV cache       │
└─────────────────────────────────────────────────────────┘
```

### 请求生命周期

```
Client                 FastAPI          Python Qwen2        C++ Worker
  │                       │                   │                  │
  │ POST /v1/chat/        │                   │                  │
  │ completions           │                   │                  │
  ├──────────────────────►│                   │                  │
  │                       │ generate_async()  │                  │
  │                       ├──────────────────►│                  │
  │                       │                   │ request(token)   │
  │                       │                   ├─────────────────►│
  │                       │                   │ condvar.wait()   │
  │                       │                   │ (线程休眠)        │
  │                       │                   │                  │
  │                       │                   │     schedule()   │
  │                       │                   │     forward()    │
  │                       │                   │     notify_all() │
  │                       │                   │◄─────────────────│
  │                       │                   │ return token     │
  │  SSE: data: {...}     │◄──────────────────│                  │
  │◄──────────────────────│                   │                  │
  │                       │                   │ request(token+1) │
  │                       │                   ├─────────────────►│
  │                       │                   │  ...循环...      │
  │                       │                   │                  │
  │  SSE: data: [DONE]    │                   │                  │
  │◄──────────────────────│                   │                  │
```

---

## 3. C++ 后端改动

### 3.1 Sequence 改造

**文件:** `src/sequence/sequence.hpp` / `src/sequence/sequence.cpp`

新增字段：
```
_token_mutex          std::mutex                  // 保护 token 状态
_token_cv             std::condition_variable     // token 更新时唤醒
_cancelled            bool = false                // 取消标志
_top_k                int                         // 构造时固化
_top_p                float                       // 构造时固化
_temperature          float                       // 构造时固化
_last_logits          tensor_t                    // 上一步 logits（logprobs 用）
```

新增方法：
```
wait_for_token(ntoken)  → condvar wait + cancelled 检查，返回 true=有token false=cancelled
notify_token()          → condvar notify_all()
cancel()               → 设 _cancelled = true, notify_all()
```

sampling 参数（top_k/top_p/temperature）通过构造函数传入并固化，不提供 setter。`config.hpp` 中的 TOP_K/TOP_P 降级为默认值。Qwen2::forward() 做 top-k 采样时从 Sequence 读取这些参数。

### 3.2 Qwen2::request() 改造

**文件:** `src/models/qwen2.hpp` / `qwen2.cpp`

```cpp
// 签名增加 sampling 参数
int64_t request(const int64_t *token_ids, size_t ntoken,
                int64_t max_new_tokens,
                int top_k, float top_p, float temperature);
```

内部实现：
```cpp
// 改前：busy-wait
while(seq->token_num() <= ntoken) { std::this_thread::yield(); }

// 改后：真等待
if (!seq->wait_for_token(ntoken)) {
    return -2;  // cancelled
}
```

创建新 Sequence 时将 top_k/top_p/temperature 传入构造函数固化。

### 3.3 Worker Thread postprocess 改造

```cpp
// token 写入后唤醒等待线程
seq->push_token(token_id);
seq->notify_token();
```

### 3.4 Scheduler::abort()

**文件:** `src/scheduler/scheduler.hpp` / `scheduler.cpp`

```cpp
void Scheduler::abort(seq_t seq) {
    // 1. 从 _waiting 或 _running 中移除
    // 2. 释放 KV cache blocks (BlockManager::deallocate)
    // 3. seq->cancel()
}
```

### 3.5 Logprobs 支持

`forward()` 计算 logits 后保存到 `seq->_last_logits`。Python 调 `GetLogprobs` 时 C++ 对 logits 做 softmax → log → 返回 top-N 个 (token_id, logprob)。

### 3.6 新增 C API

**文件:** `include/llaisys/models/qwen2.h`

```c
// 修改：llaisysQwen2ModelInfer 增加 sampling 参数
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model,
                                int64_t *token_ids, size_t ntoken,
                                int64_t max_new_tokens,
                                int top_k, float top_p, float temperature);

// 新增：取消请求
void llaisysQwen2ModelAbort(struct LlaisysQwen2Model *model, int64_t *token_ids);

// 新增：获取上一步 logprobs
int llaisysQwen2ModelGetLogprobs(struct LlaisysQwen2Model *model,
                                  float *out_logprobs, int n_vocab,
                                  int *out_tokens, int n_tokens);
```

`Abort` 通过 token_ids 定位 Sequence（与 request/infer 一致），cancel + 从 scheduler 移除。  
`GetLogprobs` 返回上一步 logprobs 的 top-N 列表。

### 3.7 C++ 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/sequence/sequence.hpp` | 修改 | cv/mutex/cancelled/sampling_config/last_logits |
| `src/sequence/sequence.cpp` | 修改 | wait_for_token/notify_token/cancel |
| `src/scheduler/scheduler.hpp` | 修改 | abort() 声明 |
| `src/scheduler/scheduler.cpp` | 修改 | abort() 实现 |
| `src/models/qwen2.hpp` | 修改 | request 签名 + get_logprobs + abort |
| `src/models/qwen2.cpp` | 修改 | condvar wait + notify + abort + logprobs |
| `include/llaisys/models/qwen2.h` | 修改 | Abort + GetLogprobs C API |
| `src/llaisys/models/qwen2.cc` | 修改 | C API bridge |

---

## 4. Python 侧改动

### 4.1 `generate_async()` — async generator

**文件:** `python/llaisys/models/qwen2.py`

```python
async def generate_async(
    self,
    inputs: list[int],
    max_new_tokens: int = -1,
    temperature: float = 0.8,
    top_k: int = 10,
    top_p: float = 0.9,
    logprobs: bool = False,
    top_logprobs: int = 0,
) -> AsyncGenerator[dict, None]:
```

内部用 `asyncio.to_thread()` 调用 `llaisysQwen2ModelInfer`，每次返回一个 token 后 yield。C++ 层 condvar 真休眠，线程不空转。

temperature/top_p/top_k 通过 `llaisysQwen2ModelInfer` 传递到 C++，最终写入 Sequence 构造参数。C++ forward() 做 top-k 采样时从 Sequence 读取这些值。

客户端断开 → `asyncio.CancelledError` → 调 `llaisysQwen2ModelAbort()`。

### 4.2 FastAPI Routes

**文件:** `python/llaisys/server.py`（新建）

| Route | 方法 | 说明 |
|-------|------|------|
| `/v1/chat/completions` | POST | 对话补全，支持 stream |
| `/v1/completions` | POST | 文本补全（legacy） |
| `/v1/models` | GET | 模型列表 |
| `/health` | GET | 健康检查 |

`chat_completions` handler：
1. 解析 `ChatCompletionRequest` (Pydantic)
2. Messages → apply chat template → tokenize
3. 参数校验 (max_tokens, temperature, top_p, ...)
4. `stream=False` → 调 `generate()` 同步返回完整 JSON
5. `stream=True` → `StreamingResponse` + SSE event stream
6. 客户端断开 → 自动触发 abort

### 4.3 SSE 格式（OpenAI 规范）

```json
// 首个 chunk（含 role）
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "created":1720000000,"model":"qwen2-1.5b",
       "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

// 内容 chunk
data: {"id":"chatcmpl-xxx","object":"chat.completion.chunk",
       "created":1720000000,"model":"qwen2-1.5b",
       "choices":[{"index":0,"delta":{"content":"你好"},"finish_reason":null}]}

// 带 logprobs
data: {"id":"chatcmpl-xxx",...,
       "choices":[{"index":0,"delta":{"content":"好"},
       "logprobs":{"content":[{"token":"好","logprob":-0.12,
       "top_logprobs":[{"token":"好","logprob":-0.12},{"token":"的","logprob":-1.5}]}]},
       "finish_reason":null}]}

// 结束
data: {"id":"chatcmpl-xxx",...,"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}
data: [DONE]
```

### 4.4 Schemas

**文件:** `python/llaisys/schemas.py`（新建）

Pydantic models：
- `ChatCompletionRequest` — messages, model, temperature, top_p, max_tokens, stream, logprobs, top_logprobs, stop, ...
- `ChatCompletionResponse` — 非流式完整响应
- `ChatCompletionChunk` — 流式 chunk
- `ModelCard` / `ModelList` — /v1/models 响应

### 4.5 RequestManager

```python
class RequestManager:
    _active: dict[str, RequestState]

    async def create(self, req_id, ...) -> AsyncGenerator
    async def cancel(self, req_id) → abort + cleanup
```

### 4.6 Python 文件变更清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `python/llaisys/server.py` | **新建** | FastAPI app + routes |
| `python/llaisys/schemas.py` | **新建** | Pydantic 请求/响应类型 |
| `python/llaisys/models/qwen2.py` | 修改 | `generate_async()` + logprobs |
| `python/llaisys/libllaisys/models/qwen2.py` | 修改 | Abort/GetLogprobs ctypes 绑定 |
| `python/setup.py` / `pyproject.toml` | 修改 | 加 fastapi/uvicorn/sse-starlette 依赖 |

---

## 5. 依赖

**Python:**
- fastapi
- uvicorn
- sse-starlette（SSE EventSourceResponse）
- pydantic

**C++:**
- 无新增依赖（condition_variable 为标准库）

---

## 6. 测试策略

| 测试类型 | 内容 |
|----------|------|
| **单元测试** | Pydantic schema 序列化/反序列化；SSE 格式正确性 |
| **集成测试** | `/v1/chat/completions` stream=False 端到端；stream=True SSE 正确性；客户端断开后 C++ 序列是否清理 |
| **回归测试** | 确保 `test/test_infer.py --test` 正确性通过；`test/test_infer_batch.py --test` 批量推理通过 |
| **手动验证** | `curl -X POST .../v1/chat/completions` + SSE 输出对比 |

---

## 7. 风险 & 注意事项

1. **condvar 跨线程安全** — Python 的 GIL + ctypes 调用 C++ condvar.wait() 时，GIL 会在进入 C 代码前释放（ctypes 默认行为），不会阻塞 event loop 的其他 coroutine。需要验证。
2. **logprobs 内存** — `_last_logits` 是 tensor_t（shared_ptr），在下次 forward 前持有引用，注意显存占用。
3. **Abort 竞态** — abort 调用时 token 刚好到达，需要正确处理 mutex 锁顺序。
4. **现有测试兼容** — 旧的 `generate()` 同步方法保留不变（内部仍用 yield wait），新增 `generate_async()` 作为 async 替代。
