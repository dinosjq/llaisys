# max_new_tokens Propagation to C++ Sequence

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propagate `max_new_tokens` from Python `generate()` to C++ `Sequence` so `postprocess()` can free KV cache blocks when the user-specified limit is reached, fixing the block leak that causes progressive latency increase.

**Architecture:** Add `max_new_tokens` parameter to `llaisysQwen2ModelInfer()` C API, thread it through `request()` to `Sequence::set_max_token_num()`. Python default is -1 (use config `MAX_TOKEN_NUM`). Existing `postprocess()` cleanup logic (`seq->token_num() >= seq->max_token_num()`) activates automatically — zero changes to scheduler/block manager.

**Tech Stack:** C++17, CUDA 12.4, Python 3.12 ctypes

## Spec Reference

Root cause confirmed in session diagnostics: `docs/superpowers/specs/` (in-conversation, see KV cache leak investigation).

## Global Constraints

- `max_new_tokens = -1` means "use config default" (`MAX_TOKEN_NUM = 8192`)
- Existing `postprocess()` cleanup logic unchanged
- Python `generate(max_new_tokens=128)` default unchanged
- All existing tests must pass

---

### Task 1: Add set_max_token_num() to Sequence

**Files:**
- Modify: `src/sequence/sequence.hpp`

**Interfaces:**
- Produces: `void Sequence::set_max_token_num(size_t v)` — sets `_max_ntoken`

- [ ] **Step 1: Add setter declaration**

In `src/sequence/sequence.hpp`, after `size_t max_token_num();` add:

```cpp
void set_max_token_num(size_t v) { _max_ntoken = v; }
```

- [ ] **Step 2: Commit**

```bash
git add src/sequence/sequence.hpp
git commit -m "feat: add set_max_token_num() setter to Sequence"
```

---

### Task 2: Add max_new_tokens to C++ request() and C API

**Files:**
- Modify: `include/llaisys/models/qwen2.h` (C API)
- Modify: `src/llaisys/models/qwen2.cc` (C→C++ bridge)
- Modify: `src/models/qwen2.hpp` (Qwen2 class)
- Modify: `src/models/qwen2.cpp` (Qwen2 impl)

**Interfaces:**
- Consumes: `Sequence::set_max_token_num()` from Task 1
- Produces: `llaisysQwen2ModelInfer(model, token_ids, ntoken, max_new_tokens)` with new 4th arg

- [ ] **Step 1: Update C API header**

In `include/llaisys/models/qwen2.h`, change:

```c
__export int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model * model, int64_t *token_ids, size_t ntoken, int64_t max_new_tokens);
```

- [ ] **Step 2: Update C→C++ bridge**

In `src/llaisys/models/qwen2.cc`, change `llaisysQwen2ModelInfer`:

```cpp
int64_t llaisysQwen2ModelInfer(struct LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken, int64_t max_new_tokens) {
    if (!model || !model->qwen2) return -1;
    return model->qwen2->request(token_ids, ntoken, max_new_tokens);
}
```

- [ ] **Step 3: Update Qwen2 class header**

In `src/models/qwen2.hpp`, change `request()` declaration:

```cpp
int64_t request(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens);
```

- [ ] **Step 4: Update Qwen2::request() implementation**

In `src/models/qwen2.cpp`, change `request()` signature to match the header, and in the new-sequence branch:

```cpp
int64_t Qwen2::request(int64_t *token_ids, size_t ntoken, int64_t max_new_tokens){
    long long hash = 0;
    std::shared_ptr<Sequence> seq;
    {
        std::shared_lock<std::shared_mutex> read_lk(this->_hash_lock);
        for(size_t i = 0; i < ntoken; ++ i){
            hash = (hash * Prime + token_ids[i]) % mod;
            if(_hash_2_seq.count(hash)){
                seq = _hash_2_seq[hash];
            }
        }
        if(seq == nullptr){
            read_lk.unlock();
            std::unique_lock<std::shared_mutex> write_lk(this->_hash_lock);
            seq = _hash_2_seq.count(hash) ? _hash_2_seq[hash] : nullptr;
            if(seq == nullptr){
                seq = std::make_shared<Sequence>(token_ids, ntoken);
                // Set sequence max tokens: -1 → use config default, else prompt + max_new_tokens
                size_t limit = (max_new_tokens < 0)
                    ? seq->max_token_num()  // MAX_TOKEN_NUM from config
                    : (ntoken + static_cast<size_t>(max_new_tokens));
                seq->set_max_token_num(limit);
                _hash_2_seq[seq->prompt_hash()] = seq;
                this->_scheduler->add(seq);
            }
        }
    }

    while(seq->token_num() <= ntoken){
        std::this_thread::yield();
    }

    int64_t token = seq->token_ids()[ntoken];
    if(token == this->_meta.end_token){
        std::unique_lock<std::shared_mutex> write_lk(this->_hash_lock);
        this->_hash_2_seq.erase(seq->prompt_hash());
    }
    return token;
}
```

- [ ] **Step 5: Build and verify compile**

```bash
xmake && xmake install
```

Expected: `build ok`

- [ ] **Step 6: Commit**

```bash
git add include/llaisys/models/qwen2.h src/llaisys/models/qwen2.cc src/models/qwen2.hpp src/models/qwen2.cpp
git commit -m "feat: propagate max_new_tokens to Sequence for KV cache cleanup"
```

---

### Task 3: Update Python bindings

**Files:**
- Modify: `python/llaisys/libllaisys/models/qwen2.py`
- Modify: `python/llaisys/models/qwen2.py`

**Interfaces:**
- Consumes: Updated `llaisysQwen2ModelInfer` C signature from Task 2
- Produces: `Qwen2.generate()` passes `max_new_tokens` through

- [ ] **Step 1: Update ctypes argtypes**

In `python/llaisys/libllaisys/models/qwen2.py`, change:

```python
lib.llaisysQwen2ModelInfer.argtypes = [
    llaisysQwen2Model_t,
    POINTER(c_int64),
    c_size_t,
    c_int64,                          # ← new: max_new_tokens
]
lib.llaisysQwen2ModelInfer.restype = c_int64
```

- [ ] **Step 2: Update Python Qwen2.generate()**

In `python/llaisys/models/qwen2.py`, change the `generate()` method's call to pass `max_new_tokens`:

```python
MAX_NEW_TOKENS_DEFAULT = -1   # -1 means "use C++ config default"

def generate(self, inputs, max_new_tokens=None, top_k=1, top_p=0.8, temperature=0.8, callback=None):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS_DEFAULT

    tokens = list(int(t) for t in inputs)
    for step in range(abs(max_new_tokens) if max_new_tokens >= 0 else 128):
        arr = (c_int64 * len(tokens))(*tokens)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model,
                arr,
                c_size_t(len(tokens)),
                c_int64(max_new_tokens),          # ← new argument
            )
        )
        tokens.append(next_token)
        if callback is not None:
            cont = callback(next_token, step, tuple(tokens))
            if cont is False:
                break
        if next_token == int(self._meta.end_token):
            break
    return tokens
```

Note: when `max_new_tokens == -1`, `abs(-1)` = 1 which would only generate 1 token. We need to handle the loop bound properly. Since -1 means "use config default" (8192), use a large fallback:

Actually, simpler approach: when `max_new_tokens < 0`, use `self._meta.maxseq` or `128` as the loop bound (the same as the old default of `128`):

```python
MAX_NEW_TOKENS_DEFAULT = -1

def generate(self, inputs, max_new_tokens=None, top_k=1, top_p=0.8, temperature=0.8, callback=None):
    if max_new_tokens is None:
        max_new_tokens = MAX_NEW_TOKENS_DEFAULT

    _loop_bound = max_new_tokens if max_new_tokens >= 0 else 128
    tokens = list(int(t) for t in inputs)
    for step in range(_loop_bound):
        arr = (c_int64 * len(tokens))(*tokens)
        next_token = int(
            LIB_LLAISYS.llaisysQwen2ModelInfer(
                self._model, arr, c_size_t(len(tokens)), c_int64(max_new_tokens),
            )
        )
        tokens.append(next_token)
        if callback is not None:
            cont = callback(next_token, step, tuple(tokens))
            if cont is False:
                break
        if next_token == int(self._meta.end_token):
            break
    return tokens
```

- [ ] **Step 3: Install and verify Python import**

```bash
./venv/bin/pip install -e ./python/
./venv/bin/python -c "import llaisys; print('import OK')"
```

Expected: `import OK`, no AttributeError

- [ ] **Step 4: Commit**

```bash
git add python/llaisys/libllaisys/models/qwen2.py python/llaisys/models/qwen2.py
git commit -m "feat: pass max_new_tokens from Python generate() to C++"
```

---

### Task 4: Verify fix — no more block leak

**Files:**
- Modify: (none — diagnostic already in place from previous work)

- [ ] **Step 1: Run diagnostic benchmark**

```bash
./venv/bin/python -c '
import sys
sys.path.insert(0, "test/benchmark")
from infer_utils import load_llaisys_model, build_prompt_bank, llaisys_infer, _apply_chat
from transformers import AutoTokenizer

MODEL = "/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B"
prompts = build_prompt_bank(6)
tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
model = load_llaisys_model(MODEL, "nvidia")

print()
print("=== Post-Fix Block Leak Diagnostic ===")
for i, p in enumerate(prompts):
    running_before = model.running_count()
    free_before = model.free_blocks()
    
    tokens, e_us, ttft_us = llaisys_infer(p, tokenizer, model, 16, 1.0, 1, 1.0)
    input_len = len(_apply_chat(tokenizer, p))
    
    running_after = model.running_count()
    free_after = model.free_blocks()
    
    print(f"  [{i}] {p[:40]:<40s} in={input_len:>4d} out={len(tokens)-input_len:>2d}  "
          f"free: {free_before:>3d}→{free_after:<3d}  "
          f"running: {running_before}→{running_after}")

print(f"\n  After all prompts: free={model.free_blocks()} running={model.running_count()}")
model.close()
'
```

Expected: `running` stays at 0 (each sequence cleaned up after generate() exits), `free` returns to baseline after each prompt.

- [ ] **Step 2: Run existing tests**

```bash
# Operator tests
for t in test/ops/*.py; do ./venv/bin/python "$t" --device nvidia; done

# Run benchmark with test mode
./venv/bin/python test/benchmark/benchmark_infer.py --model /home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B --warmup 1 --repeat 1 --num_prompts 2 --max_step 16 --test
```

Expected: All tests pass, correctness matches.

- [ ] **Step 3: Commit any remaining changes**
