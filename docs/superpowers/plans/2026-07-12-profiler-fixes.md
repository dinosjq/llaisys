# Profiler Output Fixes — Implementation Plan

> **For agentic workers:** Inline execution for tightly-coupled 3-file change.

**Goal:** Aggregate per-operator output, add console tables, --prompt-index, prevent double-dump.

## Global Constraints
- Per-operator aggregation: sum across 28 layers, display avg_ms + count + total_ms
- Console output matches format.md
- --prompt-index defaults to -1 (run all prompts)
- Double-dump prevented via `_dumped` flag in Qwen2

---

### Task 1: profiler.cpp — aggregate + console output

**Files:** `src/profiler/profiler.cpp`

- [ ] **Step 1: Rewrite dump_json**

In `dump_json()`, before writing JSON:
1. Aggregate per-operator by name across all snapshots
2. Print console table per format.md header + operator rows
3. Write aggregated JSON

- [ ] **Step 2: Commit**

```bash
git add src/profiler/profiler.cpp
git commit -m "feat: profiler aggregate per-operator + console table output"
```

---

### Task 2: qwen2.cpp — guard against double dump

**Files:** `src/models/qwen2.cpp`

- [ ] **Step 1: Add _profiled flag**

```cpp
// In Qwen2::stop()
if (!_profiled) {
    OpProfiler::instance().dump_json("...");
    _profiled = true;
}
```

Add `bool _profiled = false;` to `Qwen2` class in qwen2.hpp.

- [ ] **Step 2: Commit**

```bash
git add src/models/qwen2.cpp src/models/qwen2.hpp
git commit -m "fix: guard against double profiler dump"
```

---

### Task 3: profile_model.py — --prompt-index

**Files:** `test/profile/profile_model.py`

- [ ] **Step 1: Add argument**

```python
parser.add_argument("--prompt-index", default=-1, type=int,
    help="Run single prompt by index (0-4), default runs all")
```

Loop:
```python
for i, prompt in enumerate(PROMPTS):
    if args.prompt_index >= 0 and i != args.prompt_index:
        continue
    ...
```

- [ ] **Step 2: Commit**

```bash
git add test/profile/profile_model.py
git commit -m "feat: --prompt-index for single-prompt profiling"
```
