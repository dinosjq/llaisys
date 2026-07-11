# Streaming TTFT Integration — Implementation Plan

> **For agentic workers:** Inline execution for tightly-coupled 2-file change.

**Goal:** Use the new `Qwen2.generate(callback=)` API to measure LLAISYS TTFT in benchmark scripts.

**Architecture:** `llaisys_infer` passes a callback to `model.generate()` that captures the first token's timestamp. `llaisys_concurrent_infer` does the same in worker threads. `format_table` removes the LLAISYS TTFT "—" placeholder.

## Global Constraints

- callback signature: `callback(token, step, tokens_tuple) -> bool`
- TTFT = time.perf_counter() at generate() entry → time.perf_counter() at step=0 callback
- format_table: LLAISYS TTFT column shows real value (not "—")

---

### Task 1: Update llaisys_infer with TTFT via callback

**Files:** `test/benchmark/infer_utils.py`

- [ ] **Step 1: Rewrite llaisys_infer**

```python
def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128,
                  top_p=0.8, top_k=50, temperature=0.8):
    input_ids = _apply_chat(tokenizer, prompt)
    ttft_us = 0.0
    first = True

    def _cb(token, step, tokens_tuple):
        nonlocal ttft_us, first
        if first:
            ttft_us = (time.perf_counter() - start) * 1e6
            first = False

    start = time.perf_counter()
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        top_k=top_k, top_p=top_p, temperature=temperature,
        callback=_cb,
    )
    elapsed_us = (time.perf_counter() - start) * 1e6
    return result, elapsed_us, ttft_us
```

- [ ] **Step 2: Verify syntax**

```bash
python -c "import ast; ast.parse(open('test/benchmark/infer_utils.py','r',encoding='utf-8').read()); print('OK')"
```

- [ ] **Step 3: Commit**

```bash
git add test/benchmark/infer_utils.py
git commit -m "feat: LLAISYS TTFT via generate(callback=)"
```

---

### Task 2: Update format_table — remove LLAISYS TTFT placeholder

**Files:** `test/benchmark/infer_utils.py`

- [ ] **Step 1: Replace "—" with real value**

In `format_table`, change:
```python
f"  {'TTFT(avg)':<{METRIC_W}} {'— ms':>{LL_W}} {f'{hf_ttft:.1f} ms':>{HF_W}}"
```
To:
```python
f"  {'TTFT(avg)':<{METRIC_W}} {f'{ll_ttft:.1f} ms':>{LL_W}} {f'{hf_ttft:.1f} ms':>{HF_W}}"
```

Where `ll_ttft` is computed from results_list or ll_stats. Add `ttft_avg_ms` to ll_stats in benchmark scripts, or compute it in format_table from per-prompt data.

- [ ] **Step 2: Verify and commit**

```bash
git add test/benchmark/infer_utils.py
git commit -m "fix: LLAISYS TTFT real value in format_table"
```

---

### Task 3: Verify end-to-end

- [ ] **Step 1: Run benchmark**

```bash
python test/benchmark/benchmark_infer.py --model <path> --warmup 1 --repeat 1 --num_prompts 3
```
Expected: LLAISYS TTFT shows real value (not "—"), HF TTFT also shows real value.
