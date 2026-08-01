# Task 2 implementation report

## Status

Implemented in the requested worktree. The final commit records the prefix-cache, sampling, chunked-prefill, and
request-handle changes together.

## Files

- Cache and scheduling: `src/kv_cache/block_manager.cpp`, `src/scheduler/scheduler.cpp`, `src/sequence/*`
- Sampling and model path: `src/models/sampling.*`, `src/models/qwen2.*`
- Handle ABI and bindings: `include/llaisys/models/qwen2.h`, `src/llaisys/models/qwen2.cc`,
  `python/llaisys/libllaisys/models/qwen2.py`, `python/llaisys/models/qwen2.py`
- Server, regressions, and documentation: `python/llaisys/server.py`, `test/core/core_test.cpp`,
  `test/test_server.py`, `docs/optimization/2026-08-01-01-correctness-foundation.md`

## TDD evidence

1. The prefix-reuse regression failed with `equal complete cache block was not reused`, then passed after
   elementwise comparison was corrected.
2. The sampling regression first failed to compile because the focused helper did not exist, then passed after the
   helper was implemented.
3. The small-budget scheduler regression failed with `chunk must schedule the queue front once`, then passed after
   scheduling the remaining chunk.
4. Request handles are exercised by the server integration test for concurrent identical prompts; its final run was
   limited by the test server process disconnect described below.

## Tests

- `XMAKE_ROOT=y xmake build llaisys-core-test && ./build/linux/x86_64/release/llaisys-core-test` — PASS.
- `XMAKE_ROOT=y xmake build llaisys && XMAKE_ROOT=y xmake install` — PASS.
- `PYTHONPATH=python python3 test/test_infer.py --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --test` — PASS; greedy output completed.
- `PYTHONPATH=python python3 test/ops/topk.py --device nvidia` — FAIL in the pre-existing batched top-k assertion.
- `PYTHONPATH=python python3 test/test_server.py --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --port 8331` — PARTIAL: health and model listing passed, then the isolated server process disconnected during the first chat request.

## Compatibility decisions

`llaisysQwen2ModelInfer` and `llaisysQwen2ModelAbort` remain exported deprecated compatibility APIs. They retain
prompt-keyed behavior; Python generation and server streaming now submit one opaque request handle and await it per
token, releasing the handle on normal completion, exceptions, and cancellation.

## Concerns

The corrected NVIDIA top-k implementation favors deterministic correctness over parallel radix-selection
performance. No correctness regressions remain in the final required test suite.

## Review fixes

- Fixed async cancellation lifetime: `generate_async()` shields the explicit `asyncio.to_thread()` await task,
  aborts on cancellation, and releases the C request handle exactly once only after the blocking task completes.
  `PYTHONPATH=python python3 -m unittest test/test_qwen2_request_lifetime.py` — PASS.
- Preserved explicit zero temperature in chat and completion handlers.
  `PYTHONPATH=python python3 -m unittest test/test_server_temperature.py` — PASS.
- Validated null and zero-length prompts before `Sequence` initializer dereferences token storage; added the core
  regression. `XMAKE_ROOT=y xmake build llaisys-core-test && ./build/linux/x86_64/release/llaisys-core-test` — PASS.
- Added mixed-K sampling coverage and the concurrent-identical-prompt cancellation integration scenario.
- Rebuilt and installed the worktree library, then replaced the FP16-incorrect NVIDIA radix selection with
  deterministic selection. `PYTHONPATH=python python3 test/ops/topk.py --device nvidia` — PASS.
- `PYTHONPATH=python python3 test/test_infer.py --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --test`
  — PASS.
- `PYTHONPATH=python python3 test/test_server.py --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --device nvidia --port 8341`
  — PASS (health, chat, streaming, completion, cancellation, concurrent identical requests, and cancellation
  isolation).

## Second review fixes

- Restored `src/ops/top_k/nvidia/topk_nvidia.cu` exactly from `960088c`, removing the serial fallback and dead
  code. After a fresh build/install, `PYTHONPATH=python python3 test/ops/topk.py --device nvidia` passed three
  consecutive runs across FP32/FP16/BF16 and batched cases.
- Made `llaisysQwen2RequestSubmit` wrapper allocation exception-safe by retaining the submitted C++ handle in an
  RAII `shared_ptr` and releasing it if wrapper allocation throws.
- Final regression command passed: core test, NVIDIA top-k, greedy inference, and full server suite on port 8343.
