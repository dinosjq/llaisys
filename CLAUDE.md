# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Install

```bash
xmake                    # compile C++ backend
xmake install            # install shared library
pip install ./python/    # install Python package (editable: pip install -e ./python/)
```

CUDA/NVIDIA build (requires CUDA toolkit):
```bash
xmake f --nv-gpu=y -cv
xmake && xmake install
```

Code formatting: uses `clang-format` (LLVM-based, 4-space indent, no column limit, `InsertBraces: true`, operators placed at line start on wrap). See `.clang-format` at repo root.

Python dependencies (`pip install ./python/` installs these): `torch>=2.4.0`, `transformers`, `accelerate`, `ml_dtypes`, `fastapi`, `uvicorn`, `sse-starlette`, `pydantic`.

## Test Commands

```bash
python test/test_runtime.py --device cpu          # runtime API tests (cpu/nvidia)
python test/test_tensor.py                        # tensor operations
python test/test_ops.py                           # full operator test suite
python test/ops/argmax.py                         # individual operator test
python test/ops/topk.py                           # top-k operator test
python test/ops/paged_attention.py --device nvidia # paged attention (NVIDIA only)
python test/test_infer.py --model <path> --test   # single-request inference correctness
python test/test_infer_batch.py --model <path> --test  # batched multi-request correctness
python test/test_infer_perf.py --model <path>     # throughput benchmark
```

GPU variants: append `--device nvidia` to any test command.
Operator tests support `--profile` for benchmarking.
Tests use shared utilities from `test/test_utils.py` (`random_tensor`, `check_equal`, `benchmark`, etc.).

Server tests:
```bash
python test/test_server.py --model <path>              # OpenAI-compatible API server tests
python test/test_server_temperature.py --model <path>  # temperature sampling tests
python test/test_qwen2_request_api.py --model <path>   # request API lifecycle tests
python test/test_qwen2_request_lifetime.py --model <path>  # request handle lifecycle
```

## Architecture

LLAISYS is an educational AI inference framework. The backend is a C++17 shared library exposing a C API (`include/llaisys/` → `__export` functions). The frontend is Python using ctypes wrappers.

### Layer Stack (bottom-up)

1. **Device abstraction** (`src/device/`, `include/llaisys/runtime.h`) — `LlaisysRuntimeAPI` function pointer table unifying device ops (malloc/free, memcpy, stream, sync). Each device type provides its own implementation. CPU uses libc; NVIDIA uses CUDA Runtime APIs.

2. **Core resource management** (`src/core/`) — Thread-local `Context` manages per-device `Runtime` objects. `Runtime` owns the allocator, stream, and API table. `Storage` is the shared underlying memory block. Call `context().setDevice(type, id)` to switch the active device before doing device work.

3. **Tensor** (`src/tensor/`) — `Tensor = TensorMeta(shape/strides/dtype) + Storage + offset`. View operations (view, permute, slice) share the same storage and only mutate meta. Data movement (`load`, `to`) is the Tensor layer's responsibility.

4. **Operators** (`src/ops/`) — Each op has a single public entry point in its `op.cpp` that checks invariants, calls `context().setDevice(...)`, then dispatches by `deviceType()` switch-case to `cpu/` or `nvidia/` implementations.

5. **KV Cache** (`src/kv_cache/`) — `BlockManager` implements paged KV-cache with fixed-size blocks. Supports reference-counted block sharing, hash-based prefix matching for cache reuse across sequences, and lazy eviction.

6. **Sequence** (`src/sequence/`) — `Sequence` tracks a single inference request's lifecycle through a state machine: `WAITING` → `RUNNING` → `FINISHED`. Stores prompt tokens, generated tokens, block IDs mapping to KV-cache blocks, and prefill/decode stage flags.

7. **Scheduler** (`src/scheduler/`) — Dual-buffer producer-consumer scheduler. `add()` (called by request threads) writes to the non-active buffer; `schedule()` (called by the inference loop) reads from the active buffer, packs sequences into batches respecting `BATCH_MAX_TOKEN_NUM` and `BATCH_MAX_SEQ_NUM` limits, supports chunked prefill, and can preemptively evict running sequences when KV-cache blocks are exhausted.

### Key configuration (`src/config.hpp`)

| Constant | Default | Purpose |
|---|---|---|
| `KV_CACHE_TOKEN_NUM` | 64 | Tokens per KV-cache block |
| `KV_CACHE_BLOCK_NUM` | 128 | Total number of KV-cache blocks |
| `TOP_K` | 10 | Top-K candidates for sampling |
| `TOP_P` | 0.9 | Cumulative probability threshold |
| `BATCH_MAX_TOKEN_NUM` | 8192 | Max tokens per batch |
| `BATCH_MAX_SEQ_NUM` | 64 | Max sequences per batch |
| `MAX_TOKEN_NUM` | 2048 | Max tokens per sequence |

### C API → C++ bridge

`include/llaisys/` declares the C API. `src/llaisys/*.cc` implements those extern "C" functions, converting C handles (`LlaisysTensor*`) to internal C++ types (`llaisys::tensor_t` = `std::shared_ptr<Tensor>`). The struct `LlaisysTensor` simply wraps a `tensor_t`.

Model C API lives in `include/llaisys/models/model.h`, bridged by `src/llaisys/models/model.cc` (`llaisysModelCreate` / `SetWeight` / Request*).

### Python layers

- `python/llaisys/libllaisys/` — ctypes bindings: loads the `.so`, declares argtypes/restype for every C API function. Sub-packages mirror the C API structure (e.g., `libllaisys/models/qwen2.py` for model bindings, `libllaisys/llaisys_types.py` for enums).
- `python/llaisys/` — Pythonic wrappers: `Tensor`, `Ops`, `RuntimeAPI`, `models.Qwen2`.

### Building (xmake)

`xmake.lua` defines a chain of static libraries:
`llaisys-utils` → `llaisys-device` (+ `llaisys-device-cpu`, optionally `llaisys-device-nvidia`) → `llaisys-core` → `llaisys-tensor` → `llaisys-ops` (+ `llaisys-ops-cpu`, optionally `llaisys-ops-nvidia`) → `llaisys` (shared lib).

The `llaisys-core` target also includes `src/core/`, `src/kv_cache/`, `src/scheduler/`, and `src/models/`. The `llaisys` shared lib target additionally includes `src/sequence/` and `src/llaisys/models/`.

`xmake/cpu.lua` and `xmake/nvidia.lua` define device-specific compilation units. The `--nv-gpu=y` xmake option gates `ENABLE_NVIDIA_API` and pulls in CUDA toolchain + cuBLAS linking.

### Adding a new operator

1. Create `src/ops/<name>/op.hpp` declaring the C++ function in `llaisys::ops` namespace
2. Create `src/ops/<name>/op.cpp` with the dispatch entry (checks + setDevice + device switch)
3. Implement in `src/ops/<name>/cpu/` and/or `src/ops/<name>/nvidia/` (`.cu` files for CUDA)
4. Declare the C API in `include/llaisys/ops.h`
5. Wire the C→C++ call in `src/llaisys/ops.cc`
6. Add ctypes bindings in `python/llaisys/libllaisys/ops.py`
7. Add the Python wrapper in `python/llaisys/ops.py`
8. Add test in `test/ops/<name>.py`

### Operator inventory

Core operators (all have CPU + NVIDIA implementations): `add`, `argmax`, `embedding`, `linear`, `rms_norm`, `rope`, `rearrange`, `swiglu`.

Attention operators:
- `self_attention` — legacy contiguous-tensor attention (CPU+NVIDIA). Used in early single-request inference. Not used by the batch inference path.
- `paged_attention` — block-based attention reading from KV-cache blocks via a block-ID table (CPU+NVIDIA). Used in the current batch inference pipeline.
- `kv_cache_move` — writes computed K/V tensors into KV-cache blocks by block-ID. **NVIDIA only** — CPU throws unsupported.

Batch operator: `top_k` — finds top-k values+indices along last dim, supports batched 2D and 1D inputs (CPU+NVIDIA).

### Model inference

The Qwen2 model implementation spans two directories:

- **`src/models/qwen2.cpp`** — The actual C++ `Qwen2` class containing the Transformer forward pass, worker loop, request-handle lifecycle, KV-cache integration, and sampling. Every submitted request owns an independent `Sequence`; reusable KV blocks are managed only by `BlockManager`.

- **`src/llaisys/models/model.cc`** — Thin C API bridge: `LlaisysModel` holds `shared_ptr<framework::Model>`; Create dispatches Qwen2/Llama; weights via `SetWeight` → `Model::set_weight`.

- **`python/llaisys/models/qwen2.py` / `llama.py`** — Load safetensors via `llaisysModelSetWeight` and drive generation through `llaisysModelRequestSubmit/Await/Abort/Release`.

The forward pass per layer: embedding → rms_norm → linear(QKV) → rope → kv_cache_move → paged_attention → linear(O) → residual add → rms_norm → linear(gate/up) → swiglu → linear(down) → residual add.

Batch inference works by concatenating all sequences' tokens into contiguous tensors and tracking cut points to split results back per-sequence. Chunked prefill splits long prompts across multiple forward passes when they exceed batch token limits.

### Key macros (defined in `src/utils/`)

- `CHECK_SAME_DEVICE(...)`, `CHECK_SAME_SHAPE(...)`, `CHECK_SAME_DTYPE(...)` — op input validation
- `ASSERT(cond, msg)` — assertion with message
- `EXCEPTION_UNSUPPORTED_DEVICE` — throws for unsupported device types

### Debugging

- `tensor->debug()` — prints tensor data to stdout. Useful for comparing intermediate values against PyTorch during model inference development.
- Test utilities in `test/test_utils.py`: `random_tensor`, `random_int_tensor`, `zero_tensor`, `arrange_tensor` (create paired PyTorch+LLAISYS tensors), `check_equal` (compare with tolerances), `benchmark` (profile ops with warmup).

### Server

`python/llaisys/server.py` provides an OpenAI-compatible API server:

```bash
python -m llaisys.server --model <path> --port 8000 --device nvidia
```

Endpoints: `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming + non-streaming), `POST /v1/completions`. The server uses FastAPI + uvicorn with SSE for streaming. It loads the tokenizer via HuggingFace `transformers` for chat template support.

### Benchmarking & Profiling

**Operator benchmarks** (`test/benchmark/ops/`): CUDA Event-timed micro-benchmarks comparing LLAISYS ops against PyTorch/CUDA baselines. Supports NCU profiling (`--use-ncu`) and batch runs (`run_all_benchmarks.py`). See `test/benchmark/README.md` for full details.

**Inference benchmarks** (`test/benchmark/`): End-to-end inference performance comparison against HuggingFace, supporting single-request serial and concurrent multi-request modes.
```bash
python test/benchmark/benchmark_infer.py --model <path> [--test]    # single-request
python test/benchmark/benchmark_batch_infer.py --model <path>       # concurrent
```

**Model profiling** (`test/profile/`): Per-operator timing breakdown across decode steps to identify bottlenecks.
```bash
python test/profile/profile_model.py --model <path> [--prompt-len 500]
```

### CI/CD

GitHub Actions (`.github/workflows/build.yaml`): builds and runs tests on push/PR for both `windows-latest` and `ubuntu-latest`. CPU-only; no GPU tests in CI. Test stages: runtime API → tensor ops → individual operator tests (add, argmax, embedding, linear, rms_norm, rope, self_attention, swiglu) → single-request inference correctness.
