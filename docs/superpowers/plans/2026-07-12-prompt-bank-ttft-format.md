# Prompt Bank + TTFT + Format — Implementation Plan

> **For agentic workers:** Use inline execution for this tightly-coupled 3-file change.

**Goal:** Replace hacky prompt construction with tiered natural-language prompt bank, add HF TTFT measurement, and implement the new per-prompt output format.

**Architecture:** `infer_utils.py` provides the new `build_prompt_bank` with short/medium/long tiers, `hf_infer` returns `(tokens, total_us, ttft_us)`, `format_table` renders per-prompt rows + footer. Both benchmark scripts use disposable warmup + unique test prompts for LLAISYS KV-cache safety.

## Global Constraints

- Prompt bank: short/medium/long tiers, ≤10 per tier, all natural language
- LLAISYS: disposable warmup prompt, no repeat (unique test prompts)
- HF TTFT: custom token-by-token loop
- LLAISYS TTFT: N/A (placeholder)
- Output: per-prompt rows (latency + speedup), footer (TTFT/TPOT/Throughput)
- summarize_run scalar API preserved
- time.perf_counter() for all timing

---

### Task 1: infer_utils.py — prompt bank + TTFT + format

**Files:** `test/benchmark/infer_utils.py`

- [ ] **Step 1: Rewrite build_prompt_bank**

```python
PROMPT_BANK = {
    "short": [
        "What is the capital of France?",
        "Define backpropagation in one sentence.",
        "Write a haiku about GPU computing.",
        "Explain what 2+2 equals.",
        "What does GPU stand for?",
        "Name three primary colors.",
        "Convert 100 Celsius to Fahrenheit.",
        "What year did World War II end?",
        "Define 'algorithm' briefly.",
        "What is the speed of light?",
    ],
    "medium": [
        "Explain how the transformer attention mechanism works. Cover Q, K, V matrices, "
        "scaled dot-product attention, and why multi-head attention matters.",
        "Compare and contrast gradient descent, stochastic gradient descent, and Adam optimizer. "
        "Include their update rules and when to use each.",
        "Describe the key differences between RNNs, LSTMs, and Transformers for sequence modeling. "
        "Focus on vanishing gradients, parallelization, and long-range dependencies.",
        "Walk through the steps of batch normalization during training. Explain why it helps "
        "with internal covariate shift and how it affects gradient flow.",
        "Explain the concept of KV caching in autoregressive language model inference. "
        "Why does it speed up decoding, and how does memory usage scale with batch size and sequence length?",
    ],
    "long": [
        """You are designing a GPU inference serving system for large language models.
        The system must handle concurrent requests with varying prompt lengths (10 to 4096 tokens)
        and generate responses up to 2048 tokens each. Key constraints: 80 GB GPU memory,
        peak throughput requirement of 1000 requests per second, and P99 latency under 500ms.

        Please analyze the following aspects:
        1. Memory planning: how would you allocate KV cache blocks? What block size and total count?
        2. Batching strategy: continuous batching vs static batching. How does chunked prefill help?
        3. Scheduling: what policy would you use to maximize throughput while meeting latency SLOs?
        4. Quantization: would FP8 or INT8 KV cache help? What are the accuracy tradeoffs?

        Provide concrete numbers and reasoning for each design choice.""",
    ],
}

PROMPT_BANK["medium"].extend([
    "Describe how PyTorch's autograd engine tracks operations and computes gradients. "
    "Include the role of the computational graph, backward hooks, and why certain operations are non-differentiable.",
    "Explain speculative decoding for LLM inference. How does a draft model help the target model "
    "generate tokens faster? What are the acceptance criteria and typical speedups?",
    "Compare Flash Attention v1, v2, and v3. How does each version improve over the previous? "
    "Focus on tiling strategies, IO complexity, and hardware utilization on A100 vs H100 GPUs.",
    "Describe the complete pipeline of loading a HuggingFace model, converting it to ONNX, "
    "and deploying it with TensorRT. Include common pitfalls and optimization opportunities.",
    "Explain RoPE (Rotary Position Embedding) in detail. How does it encode relative position? "
    "Why is it preferred over learned absolute position embeddings in modern LLMs?",
])
```

- [ ] **Step 2: Rewrite hf_infer with TTFT**

```python
def hf_infer(prompt, tokenizer, model, max_new_tokens=128,
             top_p=0.8, top_k=50, temperature=0.8):
    """Single-prompt HF inference with TTFT measurement via token-by-token loop."""
    input_ids = _apply_chat(tokenizer, prompt)
    inputs = torch.tensor([input_ids], device=model.device)
    past_key_values = None
    generated = []
    ttft_us = 0.0

    start = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                # Prefill: process all input tokens at once
                outputs = model(input_ids=inputs, use_cache=True)
            else:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature > 0 and (top_p < 1.0 or top_k > 1):
                logits = logits / temperature
                if top_k > 1:
                    topk_vals, topk_idx = torch.topk(logits, top_k, dim=-1)
                    logits = logits.masked_fill(
                        logits < topk_vals[:, -1:], float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                    cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    mask = cum_probs > top_p
                    mask[:, 1:] = mask[:, :-1].clone()
                    mask[:, 0] = False
                    logits = logits.masked_fill(
                        mask.scatter(1, sorted_idx, mask), float("-inf"))
                probs = logits.softmax(dim=-1)
                next_token_id = torch.multinomial(probs, 1)
            else:
                next_token_id = logits.argmax(dim=-1, keepdim=True)

            if step == 0:
                ttft_us = (time.perf_counter() - start) * 1e6

            token_id = next_token_id.item()
            generated.append(token_id)
            if token_id == tokenizer.eos_token_id:
                break
            next_token = next_token_id

    elapsed_us = (time.perf_counter() - start) * 1e6
    tokens = input_ids + generated
    return tokens, elapsed_us, ttft_us
```

- [ ] **Step 3: Update llaisys_infer return for TTFT placeholder**

```python
def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128,
                  top_p=0.8, top_k=50, temperature=0.8):
    input_ids = _apply_chat(tokenizer, prompt)
    start = time.perf_counter()
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        top_k=top_k, top_p=top_p, temperature=temperature,
    )
    elapsed_us = (time.perf_counter() - start) * 1e6
    return result, elapsed_us, None  # ttft_us=None (future streaming)
```

- [ ] **Step 4: Rewrite format_table per format.md**

```python
def format_table(results_list, hf_stats, ll_stats, mode="single"):
    """New output: per-prompt rows + footer with TTFT/TPOT/Throughput."""
    lines = [f"\n  {'═' * 66}"]
    hdr = (f"  {'prompt':<12s} {'input':>6s} {'output':>7s}  "
           f"{'LLAISYS(ms)':>12s}  {'HF(ms)':>12s}  {'speedup':>8s}")
    lines.append(hdr)
    lines.append(f"  {'─' * 66}")

    for i, r in enumerate(results_list):
        lbl = r.get("label", f"prompt #{i}")
        sp = (hf_stats["per_prompt"][i] / r["ll_ms"]) if r["ll_ms"] > 0 else 0
        lines.append(
            f"  {lbl:<12s} {r['input_tok']:>6d} {r['output_tok']:>7d}  "
            f"{r['ll_ms']:>12.1f}  {hf_stats['per_prompt'][i]:>12.1f}  "
            f"{sp:>7.2f}x"
        )

    lines.append(f"  {'─' * 66}")
    ll_lats = [r["ll_ms"] for r in results_list]
    hf_lats = hf_stats["per_prompt"]
    lines.append(
        f"  {'summary':<12s} {ll_stats['input_tokens']:>6d} "
        f"{ll_stats['output_tokens']:>7d}  "
        f"avg {sum(ll_lats)/len(ll_lats):.1f}  avg {sum(hf_lats)/len(hf_lats):.1f}  "
        f"{hf_stats['elapsed_s'] / ll_stats['elapsed_s']:.2f}x"
    )
    lines.append(f"  {'':>12s} {'tok':>6s} {'tok':>7s}  "
                 f"p50 {sorted(ll_lats)[len(ll_lats)//2]:.1f}  "
                 f"p50 {sorted(hf_lats)[len(hf_lats)//2]:.1f}")

    lines.append(f"  {'═' * 66}")
    lines.append(f"  TTFT(avg):   LLAISYS {'—':>8s}     HF {hf_stats.get('ttft_avg_ms', 0):.1f} ms")
    lines.append(f"  TPOT:        LLAISYS {ll_stats['tpot_ms']:.2f} ms   HF {hf_stats['tpot_ms']:.2f} ms")
    lines.append(f"  Throughput:  LLAISYS {ll_stats['throughput']:.1f} t/s  HF {hf_stats['throughput']:.1f} t/s")
    return "\n".join(lines)
```

- [ ] **Step 5: Update callers — benchmark_infer.py**

HF section: `hf_infer` now returns 3-tuple `(tokens, total_us, ttft_us)`
LLAISYS section: `llaisys_infer` returns 3-tuple `(tokens, total_us, None)`. Warmup with disposable prompt. No repeat/reload (unique prompts guarantee KV cache safety).

- [ ] **Step 6: Verify syntax + commit**

```bash
python -c "import ast; ast.parse(open('test/benchmark/infer_utils.py','r',encoding='utf-8').read()); print('OK')"
git add test/benchmark/infer_utils.py test/benchmark/benchmark_infer.py test/benchmark/benchmark_batch_infer.py
git commit -m "feat: tiered prompt bank, HF TTFT, new per-prompt output format"
```
