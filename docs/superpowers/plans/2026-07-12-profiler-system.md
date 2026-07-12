# Operator-Level Inference Profiling System — Implementation Plan (v2)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add compile-time-gated operator-level profiling to the LLAISYS inference pipeline with per-decode-step sampling. Pure C++ internal — no Python bridge.

**Architecture:** OpProfiler singleton tracks step counter through model-registered operator sequence. Event loop wraps forward() with PROFILE_BEGIN/END. PROFILE_STEP() macro in each operator entry auto-advances. CUDA events pooled, deferred sync. JSON auto-dumped at model destruction.

**Tech Stack:** C++17, CUDA events, xmake

## Global Constraints

- Compile gate: xmake `--profiling=y` → `LLAISYS_ENABLE_PROFILING`
- Macros → no-op when disabled
- Profiler core in `src/profiler/` → `llaisys-core` target
- Model registers sequence in constructor, profiler auto-dumps in destructor
- Sample points: decode steps {1, 32, 128, 256, 512, 1024}
- Zero Python/C API code needed — profiler is internal C++ mechanism

---

### Task 1: xmake integration

**Files:** `xmake.lua`

- [ ] **Step 1: Add option + target wiring**

Add after `nv-gpu` option:

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

In `llaisys-core` target, add:
```lua
add_files("src/profiler/*.cpp")
```

- [ ] **Step 2: Commit**

```bash
git add xmake.lua && git commit -m "build: add --profiling option and profiler sources"
```

---

### Task 2: Profiler core — header + implementation

**Files:** `src/profiler/profiler.hpp`, `src/profiler/profiler.cpp`

- [ ] **Step 1: profiler.hpp**

```cpp
#pragma once
#include <string>
#include <vector>
#include <cstddef>
#ifdef LLAISYS_ENABLE_PROFILING
#include <cuda_runtime.h>
#endif

namespace llaisys {

class OpProfiler {
public:
    static OpProfiler& instance();

    void register_sequence(const std::string& model, size_t layers,
        const std::vector<std::string>& per_layer_ops,
        const std::vector<std::string>& pre_ops,
        const std::vector<std::string>& post_ops);
    void set_sample_points(const std::vector<int>& points);

    void begin_forward(bool is_prefill);
    void end_forward(int decode_step);
    void step();
    void dump_json(const char* path);  // called from Qwen2::stop()

private:
    OpProfiler() = default;
    // ... internal state ...
};

} // namespace llaisys

#ifdef LLAISYS_ENABLE_PROFILING
  #define PROFILE_STEP()   llaisys::OpProfiler::instance().step()
  #define PROFILE_BEGIN(p) llaisys::OpProfiler::instance().begin_forward(p)
  #define PROFILE_END(d)   llaisys::OpProfiler::instance().end_forward(d)
#else
  #define PROFILE_STEP()   ((void)0)
  #define PROFILE_BEGIN(p) ((void)0)
  #define PROFILE_END(d)   ((void)0)
#endif
```

- [ ] **Step 2: profiler.cpp** — implements register, begin/end, step, destructor dump

- [ ] **Step 3: Commit**

```bash
mkdir -p src/profiler
git add src/profiler/profiler.hpp src/profiler/profiler.cpp
git commit -m "feat: OpProfiler singleton core"
```

---

### Task 3: Operator instrumentation — 12 ops

**Files:** `src/ops/{linear,rms_norm,rope,paged_attention,swiglu,embedding,add,argmax,top_k,rearrange,self_attention,kv_cache_move}/op.cpp`

- [ ] **Step 1: Add include + PROFILE_STEP to each**

Pattern:
```cpp
#include "../../profiler/profiler.hpp"

void linear(tensor_t out, ...) {
    PROFILE_STEP();
    // ... unchanged code ...
}
```

- [ ] **Step 2: Verify build**

```bash
xmake f --nv-gpu=y --profiling=y -cv && xmake 2>&1 | tail -3
```

- [ ] **Step 3: Commit**

```bash
git add src/ops/*/op.cpp
git commit -m "feat: PROFILE_STEP in all 12 operator entry points"
```

---

### Task 4: Model integration — Qwen2

**Files:** `src/models/qwen2.cpp`, `src/models/qwen2.hpp`

- [ ] **Step 1: Register sequence in constructor**

After `start()` call:
```cpp
#ifdef LLAISYS_ENABLE_PROFILING
    OpProfiler::instance().register_sequence("qwen2", nlayer,
        {"attn_norm", "linear:q_proj", "linear:k_proj", "linear:v_proj",
         "rope:q", "rope:k", "kv_cache_move:k", "kv_cache_move:v",
         "paged_attn", "linear:o_proj", "add:attn",
         "mlp_norm", "linear:gate_proj", "linear:up_proj",
         "swiglu", "linear:down_proj", "add:mlp"},
        {"embed"},
        {"embed:gather", "out_norm", "linear:lm_head", "top_k"});
    OpProfiler::instance().set_sample_points({1, 32, 128, 256, 512, 1024});
#endif
```

- [ ] **Step 2: Wrap forward() in event loop**

```cpp
int decode_step = 0;
while(model->_running){
    auto [seqs, is_prefill] = ...;
    if(seqs.empty()) continue;

    auto block_ids = model->prepare_block_table(seqs);
    Qwen2Pack pack = is_prefill ? model->prepare_prefill(seqs) : model->prepare_decode(seqs);

    PROFILE_BEGIN(is_prefill);
    std::vector<int64_t> token_ids = model->forward(pack, block_ids);
    PROFILE_END(is_prefill ? 0 : ++decode_step);

    model->_scheduler->postprocess(seqs, token_ids, is_prefill);
}
```

- [ ] **Step 3: Add include**

```cpp
#include "../profiler/profiler.hpp"
```

- [ ] **Step 4: Commit**

```bash
git add src/models/qwen2.cpp src/models/qwen2.hpp
git commit -m "feat: Qwen2 registers profiler sequence + event loop instrumentation"
```

---

### Task 5: Python driver script

**Files:** `test/profile/profile_model.py`

- [ ] **Step 1: Write driver**

```python
#!/usr/bin/env python3
"""Run inference to trigger profiling. Profiler auto-dumps at model destruction."""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import llaisys
from transformers import AutoTokenizer

PROMPTS = [
    "What is the capital of France?",
    "Explain how backpropagation works in neural networks.",
    "You are designing a GPU inference system. Describe the key components.",
    "Write a Python function to compute the Fibonacci sequence efficiently.",
    "Summarize the transformer architecture in three paragraphs.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", default=1024, type=int)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)

    for i, prompt in enumerate(PROMPTS):
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        input_ids = tokenizer.encode(prompt)
        tokens = model.generate(input_ids, max_new_tokens=args.max_steps)
        print(f"    Generated {len(tokens) - len(input_ids)} tokens")

    # Profiler auto-dumps on model destruction
    print("Profile auto-saved to profile_qwen2.json")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
mkdir -p test/profile
git add test/profile/profile_model.py
git commit -m "feat: inference profiler Python driver"
```

---

### Task 6: Build + verify

- [ ] **Step 1: Build with profiling**

```bash
xmake f --nv-gpu=y --profiling=y -cv && xmake && xmake install && pip install -e ./python/
```

- [ ] **Step 2: Run profiler**

```bash
python test/profile/profile_model.py --model <path> --max_steps 128
```

Expected: `profile_qwen2.json` generated with per-step per-operator timing.

- [ ] **Step 3: Verify disabled mode**

```bash
xmake f --nv-gpu=y --profiling=n -cv && xmake 2>&1 | tail -3
```
Expected: clean build, no profiler code compiled.
