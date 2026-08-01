# Qwen2 旧请求 API 清理实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 删除无人调用且会按 prompt 共享 `Sequence` 的旧 Qwen2 推理 API，使 request handle 成为唯一请求生命周期入口，并保持 KV Cache 仅由 `BlockManager` 复用。

**Architecture:** 模型 create/destroy 和内部 worker start/stop 保持不变。Python `generate()/generate_async()` 通过 C request handle 调用 C++ `submit/await/abort/release`；每个 handle 独占一个 `Sequence`，`BlockManager` 独立负责 KV block 共享。

**Tech Stack:** C++17、C ABI、Python ctypes、unittest、xmake、CUDA/NVIDIA。

## Global Constraints

- 仅在 `/home/songjq/.cursor/worktrees/llaisys-main/inference-optimization` 和分支 `opt/inference-system` 工作。
- 不修改主工作区、`target.md` 或既有推理优化计划。
- 主动中断旧 `llaisysQwen2ModelInfer/Abort` ABI，不增加兼容 shim。
- 不修改 Scheduler、BlockManager、Sampling 或后续性能优化。
- 测试先行：API 表面测试必须先因旧符号仍存在而失败。
- 使用 `/home/songjq/llaisys-main/venv/bin/python` 和 `PYTHONPATH=$PWD/python`。
- NVIDIA 构建命令使用 `XMAKE_ROOT=y`。
- 代码、测试和中文交付文档使用独立 commit；不推送。

---

### Task 1: 删除旧请求路径并固化唯一 API 表面

**Files:**
- Create: `test/test_qwen2_request_api.py`
- Modify: `include/llaisys/models/qwen2.h:41-46`
- Modify: `src/llaisys/models/qwen2.cc:37-48`
- Modify: `src/models/qwen2.hpp:7-10,90-91,117-128`
- Modify: `src/models/qwen2.cpp:200-246,293-319`
- Modify: `python/llaisys/libllaisys/models/qwen2.py:64-80`
- Modify: `python/llaisys/models/qwen2.py:289-293`

**Interfaces:**
- Removes: `llaisysQwen2ModelInfer`, `llaisysQwen2ModelAbort`, `Qwen2::request(...)`, `Qwen2::abort(int64_t *, size_t)`.
- Removes state: `_hash_2_seq`, `_hash_lock`.
- Preserves: `llaisysQwen2RequestSubmit/Await/Abort/Release`.
- Preserves: `Qwen2::submit/await/abort(const qwen2_request_t &)/release`.

- [ ] **Step 1: 编写失败的 API 表面测试**

创建 `test/test_qwen2_request_api.py`：

```python
import unittest

from llaisys.libllaisys import LIB_LLAISYS


class Qwen2RequestApiTest(unittest.TestCase):
    def test_legacy_prompt_keyed_api_is_not_exported(self):
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelInfer"))
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelAbort"))

    def test_request_handle_api_is_exported(self):
        for symbol in (
            "llaisysQwen2RequestSubmit",
            "llaisysQwen2RequestAwait",
            "llaisysQwen2RequestAbort",
            "llaisysQwen2RequestRelease",
        ):
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(LIB_LLAISYS, symbol))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: 运行测试并确认 RED**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python -m unittest test/test_qwen2_request_api.py -v
```

Expected: `test_legacy_prompt_keyed_api_is_not_exported` FAIL，错误表明
`llaisysQwen2ModelInfer` 当前仍存在；request handle 测试 PASS。

- [ ] **Step 3: 删除公共 C API 和 bridge**

从 `include/llaisys/models/qwen2.h` 删除：

```cpp
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model,
    int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
    int top_k, float top_p, float temperature);

__export void llaisysQwen2ModelAbort(struct LlaisysQwen2Model *model,
    int64_t *token_ids, size_t ntoken);
```

从 `src/llaisys/models/qwen2.cc` 删除调用 `model->qwen2->request(...)` 和
`model->qwen2->abort(token_ids, ntoken)` 的两个完整 C 函数。保留紧随其后的四个
`llaisysQwen2Request*` 函数。

- [ ] **Step 4: 删除 C++ prompt-hash Sequence 所有权**

从 `src/models/qwen2.hpp` 删除：

```cpp
std::unordered_map<long long, std::shared_ptr<Sequence>> _hash_2_seq;
std::shared_mutex _hash_lock;

int64_t request(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens,
                int top_k = TOP_K, float top_p = TOP_P, float temperature = 1.0f);
void abort(int64_t *token_ids, size_t ntoken);
```

删除不再使用的：

```cpp
#include <shared_mutex>
```

从 `src/models/qwen2.cpp` 删除 `Qwen2::request(...)` 和
`Qwen2::abort(int64_t *, size_t)` 的完整定义。不要修改
`Qwen2::submit/await/abort(const qwen2_request_t &)/release`。

- [ ] **Step 5: 删除 ctypes 旧 binding 并修正错误信息**

从 `python/llaisys/libllaisys/models/qwen2.py` 删除
`llaisysQwen2ModelInfer` 和 `llaisysQwen2ModelAbort` 的 `argtypes/restype` 注册。

在 `python/llaisys/models/qwen2.py` 中修改：

```python
if next_token == -1:
    raise RuntimeError("llaisysQwen2RequestAwait failed")
```

- [ ] **Step 6: 重建并确认 GREEN**

Run:

```bash
XMAKE_ROOT=y xmake build llaisys
XMAKE_ROOT=y xmake install
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python -m unittest test/test_qwen2_request_api.py -v
```

Expected: build/install PASS；两个 API 表面测试 PASS。

- [ ] **Step 7: 运行请求生命周期定向回归**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python -m unittest \
  test/test_qwen2_invalid_submit.py \
  test/test_qwen2_request_lifetime.py \
  test/test_server_temperature.py -v
```

Expected: 全部 PASS，且没有 import-time undefined symbol 错误。

- [ ] **Step 8: 确认活动代码无旧路径**

搜索以下名称：

```text
llaisysQwen2ModelInfer
llaisysQwen2ModelAbort
Qwen2::request
_hash_2_seq
_hash_lock
```

Expected: 只允许出现在测试旧符号名称、设计/交付历史说明中；不得出现在
`include/`、`src/` 或 `python/llaisys/` 活动代码。

- [ ] **Step 9: 提交代码和测试**

```bash
git add include/llaisys/models/qwen2.h \
  src/llaisys/models/qwen2.cc \
  src/models/qwen2.hpp \
  src/models/qwen2.cpp \
  python/llaisys/libllaisys/models/qwen2.py \
  python/llaisys/models/qwen2.py \
  test/test_qwen2_request_api.py
git commit -m "refactor: remove prompt-keyed Qwen2 request API"
```

Expected: commit 成功，变更仅包含旧 API 清理、错误信息修正和 API 表面测试。

---

### Task 2: 更新当前文档并完成回归验收

**Files:**
- Modify: `CLAUDE.md:113-125`
- Modify: `docs/optimization/2026-08-01-01-correctness-foundation.md:27-38`

**Interfaces:**
- Documents: request handle 是唯一推理入口。
- Documents: `BlockManager` 是唯一 KV Cache 复用所有者。
- Does not change runtime interfaces.

- [ ] **Step 1: 更新当前架构说明**

将 `CLAUDE.md` 的 Qwen2 推理说明改为：

```markdown
- **`src/models/qwen2.cpp`** — The actual C++ `Qwen2` class containing the Transformer
  forward pass, worker loop, request-handle lifecycle, KV-cache integration, and sampling.
  Every submitted request owns an independent `Sequence`; reusable KV blocks are managed
  only by `BlockManager`.

- **`python/llaisys/models/qwen2.py`** — Loads safetensors weights and drives generation
  through `llaisysQwen2RequestSubmit/Await/Abort/Release`.
```

保留 forward pass 和 batch inference 的现有说明。

- [ ] **Step 2: 更新中文交付报告**

在 `docs/optimization/2026-08-01-01-correctness-foundation.md` 中删除“旧 API 仍作为兼容路径保留”的描述，
明确记录后续清理 commit：

```markdown
Python 生成和服务端流式输出只使用 request handle，并在正常完成、异常和取消时释放它。
旧的 `llaisysQwen2ModelInfer/Abort` 及 prompt-hash `Sequence` 复用路径已删除；Prefix Cache
继续由 `BlockManager` 提供。
```

在限制中记录这是主动的 ABI breaking change，不提供仓库外旧调用者兼容层。

- [ ] **Step 3: 构建并运行 Core 回归**

Run:

```bash
XMAKE_ROOT=y xmake build llaisys-core-test
./build/linux/x86_64/release/llaisys-core-test
```

Expected: build PASS；core test PASS，覆盖 Prefix Cache、Sampling 和 Chunked Prefill。

- [ ] **Step 4: 运行 NVIDIA Top-K 回归**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python test/ops/topk.py --device nvidia
```

Expected: FP32、FP16、BF16、batched rows 和不同 K 全部 PASS。

- [ ] **Step 5: 运行端到端 greedy inference**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python \
  test/test_infer.py \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --test
```

Expected: 模型加载和 greedy generation PASS，不出现 request handle 泄漏、取消或符号错误。

- [ ] **Step 6: 运行完整 Server 回归**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python \
  test/test_server.py \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --device nvidia --port 8331
```

Expected: health、model listing、chat/completion、相同 prompt 并发和取消隔离全部 PASS。

- [ ] **Step 7: 再次运行 API 与生命周期测试**

Run:

```bash
PYTHONPATH=$PWD/python /home/songjq/llaisys-main/venv/bin/python -m unittest \
  test/test_qwen2_request_api.py \
  test/test_qwen2_invalid_submit.py \
  test/test_qwen2_request_lifetime.py \
  test/test_server_temperature.py -v
```

Expected: 全部 PASS。

- [ ] **Step 8: 提交文档和验收记录**

```bash
git add CLAUDE.md docs/optimization/2026-08-01-01-correctness-foundation.md
git commit -m "docs: record Qwen2 request API cleanup"
```

Expected: commit 成功；工作区干净，不包含 profile artifact 或主工作区文件。

## Self-Review

- Spec coverage：旧 C ABI、C++ prompt-hash 状态、ctypes binding、错误信息、文档和全套验证均有对应步骤。
- Type consistency：保留的 C++ abort 签名始终为 `abort(const qwen2_request_t &)`；删除的签名始终为
  `abort(int64_t *, size_t)`。
- Scope：不修改 KV Cache、Scheduler、Sampling、模型生命周期或后续性能计划。
- TDD：API 表面测试先在旧共享库上 RED，删除和重建后 GREEN。
