# Operator-Level Inference Profiling System — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add compile-time-gated operator-level profiling to the LLAISYS inference pipeline with per-decode-step sampling and JSON output.

**Architecture:** OpProfiler singleton tracks a step counter through a model-registered operator sequence. Each operator entry point calls PROFILE_STEP(). The event loop wraps forward() with begin/end. CUDA events are pooled for deferred computation. A Python driver script runs inference and dumps JSON.

**Tech Stack:** C++17, CUDA events, xmake, Python ctypes

## Global Constraints

- Compile-time gate: xmake option `--profiling=y` → define `LLAISYS_ENABLE_PROFILING`
- Macros expand to no-op when profiling disabled
- CUDA events only (CPU builds compile empty ScopedTimer)
- Profiler core in `llaisys-core` target; C API bridge in `llaisys` target
- Sample points: decode steps {1, 32, 128, 256, 512, 1024}
- Model registers operator sequence in constructor
- All naming must match existing codebase convention (snake_case, `llaisys` prefix for C API)

---

### Task 1: xmake integration — profiler option + target wiring

**Files:**
- Modify: `xmake.lua`

- [ ] **Step 1: Add profiling option**

At top of `xmake.lua`, after the existing `nv-gpu` option:

```lua
option("profiling")
    set_default(false)
    set_showmenu(true)
    set_description("Enable operator-level profiling (CUDA events)")
option_end()

if has_config("profiling") then
    add_defines("LLAISYS_ENABLE_PROFILING")
end
```

- [ ] **Step 2: Add profiler sources to llaisys-core target**

In the `llaisys-core` target section, add:

```lua
add_files("src/profiler/*.cpp")
```

- [ ] **Step 3: Add C API bridge to llaisys target**

In the `llaisys` target section, add after existing `add_files("src/llaisys/models/*.cc")`:

```lua
add_files("src/llaisys/profiler.cc")
```

- [ ] **Step 4: Commit**

```bash
git add xmake.lua
git commit -m "build: add --profiling option and profiler sources to targets"
```

---

### Task 2: Profiler core — OpProfiler + ScopedTimer

**Files:**
- Create: `src/profiler/profiler.hpp`
- Create: `src/profiler/profiler.cpp`

- [ ] **Step 1: Write profiler.hpp**

```cpp
#pragma once

#include <string>
#include <vector>
#include <cstddef>
#include <cuda_runtime.h>

namespace llaisys {

struct ProfileEntry {
    std::string op_name;
    size_t layer;
    const char* phase;      // "prefill" or "decode"
    float elapsed_ms;
};

class OpProfiler {
public:
    static OpProfiler& instance();

    // Model registration (called once at construction)
    void register_sequence(const std::vector<std::string>& seq);

    // Called in event loop
    void begin_forward(bool is_prefill);
    void end_forward(int decode_step);

    // Called by PROFILE_STEP in operator entry
    void step();

    // Output
    void dump_json(const char* path);

    // Configure
    void set_sample_points(const std::vector<int>& points);

private:
    OpProfiler() = default;
    
    struct SampleWindow {
        bool is_prefill;
        std::vector<std::pair<std::string, float>> entries;  // op_name, elapsed_ms
    };
    
    std::vector<std::string> _sequence;        // registered op names
    size_t _step_counter = 0;
    size_t _layer_counter = 0;
    int _decode_step = 0;
    bool _is_prefill = false;
    
    std::vector<std::pair<std::string, float>> _current_window; // accumulating
    std::vector<SampleWindow> _snapshots;
    std::vector<int> _sample_points;
};

// RAII timer — NO sync in destructor (deferred to end_forward)
struct ScopedTimer {
    explicit ScopedTimer();
    ~ScopedTimer();
    cudaEvent_t start, stop;
};

} // namespace llaisys

// Public macros
#ifdef LLAISYS_ENABLE_PROFILING
  #define PROFILE_STEP()  llaisys::OpProfiler::instance().step()
  #define PROFILE_BEGIN(pf) llaisys::OpProfiler::instance().begin_forward(pf)
  #define PROFILE_END(ds)   llaisys::OpProfiler::instance().end_forward(ds)
#else
  #define PROFILE_STEP()  ((void)0)
  #define PROFILE_BEGIN(pf) ((void)0)
  #define PROFILE_END(ds)   ((void)0)
#endif
```

- [ ] **Step 2: Write profiler.cpp skeleton**

```cpp
#include "profiler.hpp"
#include <fstream>
#include <nlohmann/json.hpp>  // or manual JSON
// ... implementation
```

- [ ] **Step 3: Commit**

```bash
mkdir -p src/profiler
git add src/profiler/profiler.hpp src/profiler/profiler.cpp
git commit -m "feat: OpProfiler singleton + ScopedTimer skeleton"
```

---

### Task 3: Operator instrumentation — PROFILE_STEP in 12 op entry points

**Files:**
- Modify: `src/ops/linear/op.cpp`, `src/ops/rms_norm/op.cpp`, `src/ops/rope/op.cpp`, `src/ops/paged_attention/op.cpp`, `src/ops/swiglu/op.cpp`, `src/ops/embedding/op.cpp`, `src/ops/add/op.cpp`, `src/ops/argmax/op.cpp`, `src/ops/top_k/op.cpp`, `src/ops/rearrange/op.cpp`, `src/ops/self_attention/op.cpp`, `src/ops/kv_cache_move/op.cpp`

- [ ] **Step 1: Add #include and PROFILE_STEP to each op**

Pattern for each file:

```cpp
// At top, after existing includes
#include "../../profiler/profiler.hpp"

// First line of the op function body
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    PROFILE_STEP();
    // ... existing code unchanged ...
}
```

- [ ] **Step 2: Verify each compiles with profiler enabled**

```bash
xmake f --nv-gpu=y --profiling=y -cv && xmake 2>&1 | tail -5
```

- [ ] **Step 3: Commit**

```bash
git add src/ops/*/op.cpp
git commit -m "feat: add PROFILE_STEP() to all 12 operator entry points"
```

---

### Task 4: Model integration — Qwen2 registers sequence + event loop wraps forward

**Files:**
- Modify: `src/models/qwen2.cpp`, `src/models/qwen2.hpp`

- [ ] **Step 1: Register operator sequence in Qwen2 constructor**

In Qwen2 constructor (after `start()` call), add:

```cpp
#ifdef LLAISYS_ENABLE_PROFILING
    // Register the full forward operator sequence (matches forward() exactly)
    OpProfiler::instance().register_sequence({
        "embed",
        // 28 layers:
        "attn_norm", "linear:q_proj", "linear:k_proj", "linear:v_proj",
        "rope:q", "rope:k",
        "kv_cache_move:k", "kv_cache_move:v",
        "paged_attn",
        "linear:o_proj", "add:attn",
        "mlp_norm", "linear:gate_proj", "linear:up_proj",
        "swiglu",
        "linear:down_proj", "add:mlp",
        // (repeated layer_count times via loop in register function)
        "out_norm", "linear:lm_head", "top_k",
    });
    OpProfiler::instance().set_sample_points({1, 32, 128, 256, 512, 1024});
#endif
```

- [ ] **Step 2: Add PROFILE_BEGIN/PROFILE_END in event loop**

In `src/models/qwen2.cpp`, event loop (around lines 191-204):

```cpp
auto loop = [](Qwen2 *model)->void{
    int decode_step = 0;
    while(model->_running){
        auto [seqs, is_prefill] = model->_scheduler->schedule();
        if(seqs.empty()) continue;
        
        PROFILE_BEGIN(is_prefill);
        auto block_ids = model->prepare_block_table(seqs);
        Qwen2Pack pack = is_prefill ? model->prepare_prefill(seqs) : model->prepare_decode(seqs);
        std::vector<int64_t> token_ids = model->forward(pack, block_ids);
        model->_scheduler->postprocess(seqs, token_ids, is_prefill);
        PROFILE_END(is_prefill ? 0 : ++decode_step);
    }
};
```

- [ ] **Step 3: Add include**

At top of qwen2.cpp:
```cpp
#include "../profiler/profiler.hpp"
```

- [ ] **Step 4: Commit**

```bash
git add src/models/qwen2.cpp src/models/qwen2.hpp
git commit -m "feat: Qwen2 registers profiler sequence, event loop wraps forward"
```

---

### Task 5: C API + bridge

**Files:**
- Create: `include/llaisys/profiler.h`
- Create: `src/llaisys/profiler.cc`

- [ ] **Step 1: Write C API header**

```c
// include/llaisys/profiler.h
#ifndef LLAISYS_PROFILER_H
#define LLAISYS_PROFILER_H

#include "runtime.h"

__C {
    __export void llaisysProfilerRegisterSequence(const char** names, int count);
    __export void llaisysProfilerSetSamplePoints(const int* points, int count);
    __export void llaisysProfilerDumpJson(const char* path);
}
#endif
```

- [ ] **Step 2: Write C++ bridge**

```cpp
// src/llaisys/profiler.cc
#include "../../include/llaisys/profiler.h"
#include "../profiler/profiler.hpp"

extern "C" {
    void llaisysProfilerRegisterSequence(const char** names, int count) {
        std::vector<std::string> seq;
        for (int i = 0; i < count; i++) seq.push_back(names[i]);
        llaisys::OpProfiler::instance().register_sequence(seq);
    }
    void llaisysProfilerSetSamplePoints(const int* points, int count) {
        llaisys::OpProfiler::instance().set_sample_points(
            std::vector<int>(points, points + count));
    }
    void llaisysProfilerDumpJson(const char* path) {
        llaisys::OpProfiler::instance().dump_json(path);
    }
}
```

- [ ] **Step 3: Commit**

```bash
git add include/llaisys/profiler.h src/llaisys/profiler.cc
git commit -m "feat: C API for profiler registration and dump"
```

---

### Task 6: Python bindings

**Files:**
- Create: `python/llaisys/libllaisys/profiler.py`
- Modify: `python/llaisys/libllaisys/__init__.py`

- [ ] **Step 1: Write ctypes bindings**

```python
# python/llaisys/libllaisys/profiler.py
from ctypes import c_char_p, c_int, POINTER

def load_profiler(lib):
    lib.llaisysProfilerRegisterSequence.argtypes = [POINTER(c_char_p), c_int]
    lib.llaisysProfilerRegisterSequence.restype = None
    
    lib.llaisysProfilerSetSamplePoints.argtypes = [POINTER(c_int), c_int]
    lib.llaisysProfilerSetSamplePoints.restype = None
    
    lib.llaisysProfilerDumpJson.argtypes = [c_char_p]
    lib.llaisysProfilerDumpJson.restype = None
```

- [ ] **Step 2: Wire into __init__.py**

After `load_qwen2(lib)` line, add:
```python
from .profiler import load_profiler
load_profiler(lib)
```

- [ ] **Step 3: Commit**

```bash
git add python/llaisys/libllaisys/profiler.py python/llaisys/libllaisys/__init__.py
git commit -m "feat: Python ctypes bindings for profiler C API"
```

---

### Task 7: Python driver script

**Files:**
- Create: `test/profile/profile_model.py`

- [ ] **Step 1: Write driver**

```python
#!/usr/bin/env python3
"""Inference profiler — samples operator timing at multiple decode lengths."""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import llaisys
from transformers import AutoTokenizer

SAMPLE_POINTS = [1, 32, 128, 256, 512, 1024]
PROMPTS = [
    "What is the capital of France?",
    "Explain how backpropagation works in neural networks.",
    "You are designing a GPU inference system. Describe the key components: KV cache, batching, and scheduling.",
    "Write a Python function to compute the Fibonacci sequence efficiently.",
    "Summarize the transformer architecture in three paragraphs.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--max_steps", default=1024, type=int)
    parser.add_argument("--output", default="test/profile/results", type=str)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)

    os.makedirs(args.output, exist_ok=True)

    for i, prompt in enumerate(PROMPTS):
        print(f"\n  Prompt {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        input_ids = tokenizer.encode(prompt)
        tokens = model.generate(input_ids, max_new_tokens=args.max_steps)
        print(f"    Generated {len(tokens) - len(input_ids)} tokens")

    # Dump profiler data
    json_path = os.path.join(args.output, "profile.json")
    llaisys.libllaisys.LIB_LLAISYS.llaisysProfilerDumpJson(
        json_path.encode("utf-8"))
    print(f"\n  Profile saved: {json_path}")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add test/profile/profile_model.py
git commit -m "feat: inference profiler Python driver"
```

---

### Task 8: Build + test

- [ ] **Step 1: Build with profiling enabled**

```bash
xmake f --nv-gpu=y --profiling=y -cv && xmake && xmake install && pip install -e ./python/
```

- [ ] **Step 2: Run profiler**

```bash
python test/profile/profile_model.py --model <path> --max_steps 128
```

Expected: `test/profile/results/profile.json` with per-sample-point per-operator timing data.

- [ ] **Step 3: Verify profiling disabled has zero overhead**

```bash
xmake f --nv-gpu=y --profiling=n -cv && xmake 2>&1 | grep -i "profiling\|error"
```
Expected: clean build, no error.
