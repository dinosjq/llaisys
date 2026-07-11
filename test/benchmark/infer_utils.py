"""Shared utilities for inference benchmarks."""

import gc
import json
import os
import sys
import threading
import time
import warnings

# Allow import from test/ directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

import llaisys
from test.test_utils import torch_device, llaisys_device

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated!")


# ── Model Loading ─────────────────────────────────────────────────

def load_hf_model(model_path=None, device_name="nvidia"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path and os.path.isdir(model_path):
        print(f"Loading HF model from local: {model_path}")
    else:
        print(f"Downloading HF model: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name), trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model, model_path


def load_llaisys_model(model_path, device_name):
    return llaisys.models.Qwen2(model_path, llaisys_device(device_name))


# ── Prompt Construction ──────────────────────────────────────────

# ── Prompt Bank (short / medium / long tiers) ────────────────────

_PROMPT_BANK = {
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
        "Why does it speed up decoding, and how does memory usage scale with batch size "
        "and sequence length?",
        "Describe how PyTorch's autograd engine tracks operations and computes gradients. "
        "Include the role of the computational graph, backward hooks, and why certain "
        "operations are non-differentiable.",
        "Explain speculative decoding for LLM inference. How does a draft model help the "
        "target model generate tokens faster? What are typical speedups?",
        "Compare Flash Attention v1, v2, and v3. How does each version improve over the "
        "previous? Focus on tiling strategies and IO complexity.",
        "Describe the complete pipeline of loading a HuggingFace model, converting it to "
        "ONNX, and deploying it with TensorRT. Include common pitfalls.",
        "Explain RoPE (Rotary Position Embedding) in detail. How does it encode relative "
        "position? Why is it preferred over learned absolute position embeddings?",
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
        """You are evaluating three different GPU architectures for LLM inference deployment:
NVIDIA A100 (80GB), H100 (80GB), and B200. A single model instance serves 7B-parameter
models with FP16 weights and FP8 KV cache. The workload mix is 40% short prompts
(<50 tokens) and 60% long prompts (500-2000 tokens), with output lengths averaging
200 tokens.

For each GPU architecture, analyze:
1. Peak throughput in tokens/second under continuous batching with batch size 64.
2. Memory utilization: weight memory, KV cache memory, and activation memory breakdown.
3. Prefill vs decode bottlenecks: at what sequence length does each become the bottleneck?
4. Cost per million tokens: assume A100=$1.20/hr, H100=$2.50/hr, B200=$4.00/hr.
5. If you could only deploy on ONE architecture, which would you choose and why?

Support your analysis with computed TFLOPS, memory bandwidth numbers, and arithmetic
intensity calculations where applicable.""",
    ],
}


def build_prompt_bank(num_prompts):
    """Sample evenly from short/medium/long tiers to ensure input-length diversity."""
    tiers = ["short", "medium", "long"]
    prompts = []
    for i in range(num_prompts):
        tier = tiers[i % len(tiers)]
        pool = _PROMPT_BANK[tier]
        prompts.append(pool[i // len(tiers) % len(pool)])
    return prompts


# ── Single-Prompt Inference (CUDA Event timing) ──────────────────

def _apply_chat(tokenizer, prompt):
    content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True, tokenize=False,
    )
    return tokenizer.encode(content)


def hf_infer(prompt, tokenizer, model, max_new_tokens=128,
             top_p=0.8, top_k=50, temperature=0.8):
    """Single-prompt HF inference with TTFT via token-by-token loop."""
    input_ids = _apply_chat(tokenizer, prompt)
    inputs = torch.tensor([input_ids], device=model.device)
    past_key_values = None
    generated = []
    ttft_us = 0.0
    next_token = None

    start = time.perf_counter()
    with torch.no_grad():
        for step in range(max_new_tokens):
            if past_key_values is None:
                outputs = model(input_ids=inputs, use_cache=True)
            else:
                outputs = model(
                    input_ids=next_token,
                    past_key_values=past_key_values, use_cache=True,
                )
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            if temperature > 0 and (top_p < 1.0 or top_k > 1):
                logits = logits / max(temperature, 1e-6)
                if top_k > 1:
                    topk_vals, topk_idx = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1)
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


# ── Batch / Concurrent Inference ─────────────────────────────────

def hf_batch_infer(prompts, tokenizer, model, max_new_tokens=128,
                   top_p=0.8, top_k=50, temperature=0.8):
    """Padded batch forward pass.  Returns (results, elapsed_us)."""
    contents = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False,
        ) for p in prompts
    ]
    enc = tokenizer(contents, return_tensors="pt", padding=True,
                    truncation=True).to(model.device)
    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(
            **enc, max_new_tokens=max_new_tokens, top_k=top_k,
            top_p=top_p, temperature=temperature, use_cache=True,
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0

    input_lens = enc.attention_mask.sum(dim=1).tolist()
    max_input_len = enc.input_ids.shape[1]
    results = []
    for idx in range(len(prompts)):
        input_ids = outputs[idx, :input_lens[idx]].tolist()
        full = outputs[idx].tolist()
        gen_part = full[max_input_len:]
        while gen_part and gen_part[-1] == tokenizer.pad_token_id:
            gen_part.pop()
        results.append((input_ids, input_ids + gen_part))
    return results, elapsed_us


def llaisys_concurrent_infer(prompts, tokenizer, model, max_new_tokens=128,
                             top_p=0.8, top_k=50, temperature=0.8):
    """Threaded concurrent requests. Returns (results, elapsed_s)."""
    results = [None] * len(prompts)

    def _worker(idx, prompt):
        results[idx] = llaisys_infer(
            prompt, tokenizer, model, max_new_tokens=max_new_tokens,
            top_p=top_p, top_k=top_k, temperature=temperature,
        )

    threads = []
    start = time.perf_counter()
    for i, p in enumerate(prompts):
        t = threading.Thread(target=_worker, args=(i, p))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    elapsed_s = time.perf_counter() - start
    torch.cuda.synchronize()
    return results, elapsed_s


# ── Metrics ──────────────────────────────────────────────────────

def summarize_run(name, elapsed_s, num_prompts,
                  total_input_tokens, total_output_tokens,
                  latencies_us=None):
    """Compute aggregate metrics from pre-computed scalars.

    *latencies_us* is optional — when provided (single-request mode with
    CUDA Event timing), P50/P90 are computed.  When None (concurrent
    mode), latency percentiles are omitted.
    """
    if total_output_tokens == 0:
        total_output_tokens = 1
    tpot_ms = (elapsed_s * 1000.0 / total_output_tokens)
    throughput = total_output_tokens / elapsed_s if elapsed_s > 0 else 0
    result = {
        "name": name,
        "elapsed_s": elapsed_s,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tpot_ms": tpot_ms,
        "throughput": throughput,
    }
    if latencies_us:
        sorted_lat = sorted(latencies_us)
        n = len(sorted_lat)
        result["lat_avg"] = sum(latencies_us) / n
        result["lat_p50"] = sorted_lat[n // 2]
        result["lat_p90"] = sorted_lat[max(int(n * 0.9) - 1, 0)]
    return result


def print_speedup(hf_stats, ll_stats):
    if ll_stats["elapsed_s"] <= 0:
        return
    s_wall = hf_stats["elapsed_s"] / ll_stats["elapsed_s"]
    s_tpot = hf_stats["tpot_ms"] / ll_stats["tpot_ms"] if ll_stats["tpot_ms"] > 0 else 0
    print(f"\n  Wall-clock speedup: {s_wall:.2f}x")
    print(f"  TPOT improvement:   {s_tpot:.2f}x")


def format_table(results_list, hf_stats, ll_stats, mode="single"):
    """Per-prompt rows + footer with TTFT/TPOT/Throughput."""
    W = 68
    lines = [f"\n  {'═' * W}"]
    hdr = (f"  {'prompt':<13s} {'input':>6s} {'output':>7s}  "
           f"{'LLAISYS(ms)':>12s}  {'HF(ms)':>12s}  {'speedup':>8s}")
    lines.append(hdr)
    lines.append(f"  {'─' * W}")

    for i, r in enumerate(results_list):
        ll_ms = r["ll_ms"] / 1000.0
        hf_ms = hf_stats["per_prompt"][i] / 1000.0 if i < len(hf_stats["per_prompt"]) else 0
        sp = hf_ms / ll_ms if ll_ms > 0 else 0
        lines.append(
            f"  {r['label']:<13s} {r['input_tok']:>6d} {r['output_tok']:>7d}  "
            f"{ll_ms:>12.1f}  {hf_ms:>12.1f}  {sp:>7.2f}x"
        )

    lines.append(f"  {'─' * W}")
    ll_lats = [r["ll_ms"] / 1000.0 for r in results_list]
    hf_lats = [hf_stats["per_prompt"][i] / 1000.0
               for i in range(len(results_list))]
    lines.append(
        f"  {'summary':<13s} {ll_stats['input_tokens']:>6d} {ll_stats['output_tokens']:>7d}  "
        f"avg {sum(ll_lats)/len(ll_lats):.1f}  avg {sum(hf_lats)/len(hf_lats):.1f}  "
        f"{hf_stats['elapsed_s'] / ll_stats['elapsed_s']:.2f}x"
    )
    lines.append(
        f"  {'':>13s} {'tok':>6s} {'tok':>7s}  "
        f"p50 {sorted(ll_lats)[len(ll_lats)//2]:.1f}  "
        f"p50 {sorted(hf_lats)[len(hf_lats)//2]:.1f}"
    )

    lines.append(f"  {'═' * W}")
    hf_ttft = hf_stats.get("ttft_avg_ms", 0)
    lines.append(f"  TTFT(avg):   LLAISYS {'—':>8s}     HF {hf_ttft:.1f} ms")
    lines.append(f"  TPOT:        LLAISYS {ll_stats['tpot_ms']:.2f} ms   "
                 f"HF {hf_stats['tpot_ms']:.2f} ms")
    lines.append(f"  Throughput:  LLAISYS {ll_stats['throughput']:.1f} t/s  "
                 f"HF {hf_stats['throughput']:.1f} t/s")
    return "\n".join(lines)


def save_json(hf_stats, ll_stats, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "unknown"
    payload = {
        "environment": {"gpu": gpu, "cuda": torch.version.cuda, "timestamp": ts},
        "config": config,
        "results": {
            "hf": hf_stats,
            "llaisys": ll_stats,
            "speedup": {
                "wall_clock": hf_stats["elapsed_s"] / ll_stats["elapsed_s"]
                if ll_stats["elapsed_s"] > 0 else 0,
                "tpot": hf_stats["tpot_ms"] / ll_stats["tpot_ms"]
                if ll_stats["tpot_ms"] > 0 else 0,
            }
        }
    }
    path = os.path.join(run_dir, "results.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON saved: {path}")
    return path
