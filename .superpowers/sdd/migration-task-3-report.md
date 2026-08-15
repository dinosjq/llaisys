# Migration Task 3 Report — Qwen2 legacy vs layer greedy parity

**Date:** 2026-08-16  
**Worktree:** `/home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework`  
**Branch:** `feat/model-layer-framework`  
**Base:** `1ee414f`  
**Commit:** (see git log) `test: enforce qwen2 legacy vs layer greedy parity`

## Summary

Hard gate: real-model greedy generation (`temperature=0`, `top_k=1`) with `LLAISYS_QWEN2_LAYER_FORWARD=0` vs `=1` must emit identical token sequences for ≥32 new steps. Gate **PASSED**.

## Fix required for gate (teardown SEGV)

Layer path generated correct tokens but crashed on `llaisysQwen2ModelDestroy` (`rc=-11`). Cause: production `run_qwen2_layer_stack` always ran CPU `stable_capture` on the worker thread; host `Storage` held a reference to that thread’s `Runtime`. After worker join, thread_local Context/Runtime was gone → destructor SEGV.

**Fix:** gate layer-0 capture behind `ModelContext::enable_layer0_capture` (default `false`); NVIDIA capture unit test enables it; clear capture slots in `~Qwen2` before joining the worker.

## Build / install

```bash
cd /home/songjq/.cursor/worktrees/llaisys-main/model-layer-framework
export XMAKE_ROOT=y
xmake f --nv-gpu=y -y && xmake && xmake install
# Python: use venv + PYTHONPATH to this tree’s package (editable pip blocked by egg-info perms)
source ~/llaisys-main/venv/bin/activate
PYTHONPATH=./python:$PYTHONPATH python -c "import llaisys; print(llaisys.__file__)"
# → .../model-layer-framework/python/llaisys/__init__.py
# libllaisys.so installed under python/llaisys/libllaisys/
```

## Verification

```bash
XMAKE_ROOT=y xmake run llaisys-qwen2-dual-path-nvidia-test
→ qwen2 dual path NVIDIA test passed

XMAKE_ROOT=y xmake run llaisys-model-layer-nvidia-test
→ qwen2 NVIDIA layer stack capture test passed

PYTHONPATH=./python:$PYTHONPATH python test/test_qwen2_layer_parity.py \
  --device nvidia \
  --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --max_steps 32 --skip-hf
```

### Hard-gate evidence (token lists equal)

```text
prompt_tokens (10): [151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198]

legacy tokens (42): [151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894]

layer  tokens (42): [151646, 151646, 151644, 15191, 525, 498, 30, 151645, 151648, 198, 91786, 0, 358, 2776, 18183, 39350, 10911, 16, 11, 458, 20443, 11229, 17847, 3465, 553, 18183, 39350, 13, 358, 2776, 518, 697, 2473, 323, 1035, 387, 33972, 311, 7789, 498, 448, 894]

HARD GATE PASSED: legacy and layer greedy token sequences are identical
  equal_token_count=42
```

## Files

- `test/test_qwen2_layer_parity.py` — subprocess dual-path greedy hard gate
- `src/models/framework/model_context.hpp` — `enable_layer0_capture`
- `src/models/layers/qwen2/stack.cpp` — gate stable_capture
- `src/models/qwen2.cpp` — clear captures before worker join
- `test/models/qwen2_layer_nvidia_test.cpp` — enable capture for unit test

## Concerns / follow-ups

- Layer path is slower in this run (~19.5s vs ~6.9s legacy) for the same 32-step gate; not a correctness failure.
- Host `Storage` still binds to the allocating thread’s `Runtime`; production must not keep worker-thread captures across join (flag stays off by default).
- Default forward remains legacy until a later flip task.
