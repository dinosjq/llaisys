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

def build_prompt_bank(num_prompts, context_repeat):
    base_prompts = [
        "Explain what batch inference is and why it matters for serving large language models.",
        "Write a concise comparison between prefill and decode phases in autoregressive generation.",
        "Summarize the tradeoffs between latency, throughput, and batch size in inference systems.",
        "Give a step-by-step explanation of how KV cache helps speed up transformer decoding.",
        "Describe three common bottlenecks in GPU inference pipelines and how to measure them.",
        "Draft a short checklist for validating a new model backend before releasing it to users.",
        "What are the risks of measuring performance with too few prompts or too few generated tokens?",
        "Produce a short technical note about concurrency, scheduling, and tail latency in model serving.",
    ]
    extra_contexts = [
        "Use concrete terms and keep the answer focused on engineering tradeoffs.",
        "Include at least one practical example or metric in the response.",
        "Mention how the answer would change under CPU and GPU execution.",
        "Keep the output dense enough to exercise generation cost, not just a one-line response.",
    ]
    prompts = []
    for idx in range(num_prompts):
        topic = base_prompts[idx % len(base_prompts)]
        extra = extra_contexts[idx % len(extra_contexts)]
        context_lines = [
            f"Context line {ri + 1}: this benchmark should stress input handling, decoding, and batching behavior."
            for ri in range(context_repeat)
        ]
        prompt = f"{topic}\n\n{extra}"
        if context_lines:
            prompt += "\n\n" + "\n".join(context_lines)
        prompts.append(prompt)
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
    input_ids = _apply_chat(tokenizer, prompt)
    inputs = torch.tensor([input_ids], device=model.device)
    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            inputs, max_new_tokens=max_new_tokens, top_k=top_k,
            top_p=top_p, temperature=temperature, use_cache=True,
        )
    torch.cuda.synchronize()
    elapsed_us = (time.perf_counter() - start) * 1e6
    tokens = outputs[0].tolist()
    return tokens, elapsed_us


def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128,
                  top_p=0.8, top_k=50, temperature=0.8):
    input_ids = _apply_chat(tokenizer, prompt)
    start = time.perf_counter()
    result = model.generate(
        input_ids, max_new_tokens=max_new_tokens,
        top_k=top_k, top_p=top_p, temperature=temperature,
    )
    elapsed_us = (time.perf_counter() - start) * 1e6
    return result, elapsed_us


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


def format_table(hf_stats, ll_stats, mode="single"):
    """Unified terminal table output."""
    def _fmt(v, unit):
        return f"{v:.1f} {unit}"

    lines = [f"\n  ┌{'─' * 66}┐"]

    rows = [("TPOT", "ms", hf_stats["tpot_ms"], ll_stats["tpot_ms"]),
            ("Throughput", "tok/s", hf_stats["throughput"], ll_stats["throughput"])]
    # Latencies stored in microseconds — display as ms
    if "lat_avg" in hf_stats and "lat_avg" in ll_stats:
        rows.insert(0, ("Latency(p50)", "ms",
                        hf_stats["lat_p50"] / 1000, ll_stats["lat_p50"] / 1000))
        rows.insert(0, ("Latency(p90)", "ms",
                        hf_stats["lat_p90"] / 1000, ll_stats["lat_p90"] / 1000))
        rows.insert(0, ("Latency(avg)", "ms",
                        hf_stats["lat_avg"] / 1000, ll_stats["lat_avg"] / 1000))
    if mode == "concurrent":
        rows.insert(0, ("Total elapsed", "s",
                        hf_stats["elapsed_s"], ll_stats["elapsed_s"]))

    for metric, unit, hf_v, ll_v in rows:
        sp = hf_v / ll_v if ll_v > 0 else 0
        lines.append(
            f"  │ {metric:<18s} {_fmt(hf_v, unit):>12s}  {_fmt(ll_v, unit):>12s}  "
            f"{sp:5.2f}x │"
        )
    lines.append(f"  └{'─' * 66}┘")
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
