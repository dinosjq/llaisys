"""Shared utilities for inference benchmarks (symmetric HF ↔ LLAISYS).

Inputs are built to exact token lengths at runtime and shared across backends.
"""

from __future__ import annotations

import gc
import json
import os
import sys
import threading
import time
import warnings
from typing import Sequence

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

_SEED_TEXT = (
    "The quick brown fox jumps over the lazy dog. "
    "Machine learning is a subset of artificial intelligence that enables systems "
    "to learn and improve from experience without being explicitly programmed. "
    "Deep learning uses neural networks with many layers to model complex patterns. "
)


# ── Model loading ─────────────────────────────────────────────────

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


def load_llaisys_model(model_path, device_name, *, quantize: bool = False):
    return llaisys.models.load_causal_lm(
        model_path,
        llaisys_device(device_name),
        quantize=quantize,
    )


# ── Exact-length input construction (token units) ─────────────────

def build_input_ids(
    tokenizer,
    target_len: int,
    *,
    seed_offset: int = 0,
    use_chat_template: bool = True,
) -> list[int]:
    """Build a single prompt with exactly ``target_len`` tokens."""
    if target_len < 1:
        raise ValueError("target_len must be >= 1")

    # Vary seed slightly so multi-prompt batches are not identical prefixes.
    prefix = f"[bench-{seed_offset}] "
    text = prefix + _SEED_TEXT
    ids: list[int] = []
    for _ in range(64):
        if use_chat_template:
            try:
                content = tokenizer.apply_chat_template(
                    conversation=[{"role": "user", "content": text}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
            except Exception:
                content = text
            ids = tokenizer.encode(content, add_special_tokens=False)
        else:
            ids = tokenizer.encode(text, add_special_tokens=True)

        if len(ids) >= target_len:
            return ids[:target_len]
        text = text + " " + _SEED_TEXT

    # Still short: tile existing tokens.
    if not ids:
        ids = tokenizer.encode(_SEED_TEXT, add_special_tokens=True) or [0]
    while len(ids) < target_len:
        ids.extend(ids)
    return ids[:target_len]


def build_workload(
    tokenizer,
    input_lens: Sequence[int],
    prompts_per_len: int = 2,
    *,
    use_chat_template: bool = True,
) -> list[dict]:
    """Return shared cases: [{label, input_len, input_ids}, ...]."""
    cases = []
    for length in input_lens:
        for i in range(prompts_per_len):
            ids = build_input_ids(
                tokenizer,
                int(length),
                seed_offset=len(cases),
                use_chat_template=use_chat_template,
            )
            assert len(ids) == int(length), (len(ids), length)
            cases.append({
                "label": f"L{length}_{i}",
                "input_len": int(length),
                "input_ids": ids,
            })
    return cases


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def make_warmup_ids(input_ids: list[int], vocab_size: int) -> list[int]:
    """Same length, different first token → avoid prefix-cache hit."""
    out = list(input_ids)
    out[0] = (out[0] + 1) % max(vocab_size, 2)
    if out[0] == input_ids[0]:
        out[0] = (out[0] + 1) % max(vocab_size, 2)
    return out


# ── HF generate helpers ───────────────────────────────────────────

def _hf_gen_kwargs(max_new_tokens, top_p, top_k, temperature, force_max_tokens):
    greedy = temperature <= 0 or (top_k <= 1 and top_p >= 1.0)
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": not greedy,
        "use_cache": True,
    }
    if force_max_tokens:
        kwargs["min_new_tokens"] = max_new_tokens
    if greedy:
        return kwargs
    kwargs["temperature"] = max(temperature, 1e-5)
    kwargs["top_k"] = top_k
    kwargs["top_p"] = top_p
    return kwargs


def hf_generate_single(
    input_ids: list[int],
    tokenizer,
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    *,
    force_max_tokens: bool = False,
) -> tuple[list[int], float, float]:
    """One prompt via ``model.generate``. Returns (full_ids, elapsed_us, ttft_us≈0)."""
    device = model.device
    inputs = torch.tensor([input_ids], device=device)
    attn = torch.ones_like(inputs)
    kwargs = _hf_gen_kwargs(max_new_tokens, top_p, top_k, temperature, force_max_tokens)
    if tokenizer.pad_token_id is not None:
        kwargs["pad_token_id"] = tokenizer.pad_token_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(input_ids=inputs, attention_mask=attn, **kwargs)
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0
    full = outputs[0].tolist()
    # TTFT not available from one-shot generate; leave 0.
    return full, elapsed_us, 0.0


def hf_generate_serial(
    cases: list[dict],
    tokenizer,
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    *,
    force_max_tokens: bool = False,
) -> tuple[list[tuple[list[int], float]], float]:
    """Run cases one-by-one; returns (per_case results, wall_s)."""
    results = []
    t0 = time.perf_counter()
    for case in cases:
        full, e_us, _ = hf_generate_single(
            case["input_ids"], tokenizer, model,
            max_new_tokens, top_p, top_k, temperature,
            force_max_tokens=force_max_tokens,
        )
        results.append((full, e_us))
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t0
    return results, wall_s


def hf_generate_padded(
    cases: list[dict],
    tokenizer,
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
    *,
    force_max_tokens: bool = False,
) -> tuple[list[tuple[list[int], list[int]]], float]:
    """Static padded batch. Returns ([(input_ids, full_ids), ...], elapsed_us)."""
    max_len = max(c["input_len"] for c in cases)
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = 0
    batch = []
    attn = []
    for c in cases:
        ids = c["input_ids"]
        pad_n = max_len - len(ids)
        # left-pad for decoder-only
        batch.append([pad_id] * pad_n + ids)
        attn.append([0] * pad_n + [1] * len(ids))
    input_ids = torch.tensor(batch, device=model.device)
    attention_mask = torch.tensor(attn, device=model.device)
    kwargs = _hf_gen_kwargs(max_new_tokens, top_p, top_k, temperature, force_max_tokens)
    kwargs["pad_token_id"] = pad_id
    if tokenizer.eos_token_id is not None:
        kwargs["eos_token_id"] = tokenizer.eos_token_id

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **kwargs,
        )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0

    results = []
    for i, c in enumerate(cases):
        full = outputs[i].tolist()
        # drop left pads from output
        pad_n = max_len - c["input_len"]
        full = full[pad_n:]
        while full and full[-1] == pad_id:
            full.pop()
        results.append((c["input_ids"], full))
    return results, elapsed_us


# ── LLAISYS generate helpers ──────────────────────────────────────

def llaisys_generate_single(
    input_ids: list[int],
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> tuple[list[int], float, float]:
    ttft_us = 0.0
    first = True

    def _cb(token, step, tokens_tuple):
        nonlocal ttft_us, first
        if first:
            ttft_us = (time.perf_counter() - t0) * 1e6
            first = False

    t0 = time.perf_counter()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        callback=_cb,
    )
    end.record()
    torch.cuda.synchronize()
    elapsed_us = start.elapsed_time(end) * 1000.0
    return result, elapsed_us, ttft_us


def llaisys_generate_serial(
    cases: list[dict],
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> tuple[list[tuple[list[int], float, float]], float]:
    results = []
    t0 = time.perf_counter()
    for case in cases:
        full, e_us, ttft = llaisys_generate_single(
            case["input_ids"], model, max_new_tokens, top_p, top_k, temperature,
        )
        results.append((full, e_us, ttft))
    torch.cuda.synchronize()
    return results, time.perf_counter() - t0


def llaisys_generate_concurrent(
    cases: list[dict],
    model,
    max_new_tokens: int,
    top_p: float,
    top_k: int,
    temperature: float,
) -> tuple[list[tuple[list[int], float, float]], float]:
    results: list = [None] * len(cases)

    def _worker(idx, ids):
        full, e_us, ttft = llaisys_generate_single(
            ids, model, max_new_tokens, top_p, top_k, temperature,
        )
        results[idx] = (full, e_us, ttft)

    threads = []
    t0 = time.perf_counter()
    for i, case in enumerate(cases):
        t = threading.Thread(target=_worker, args=(i, case["input_ids"]))
        t.start()
        threads.append(t)
    for t in threads:
        t.join()
    torch.cuda.synchronize()
    return results, time.perf_counter() - t0


# ── Metrics / reporting ───────────────────────────────────────────

def count_new_tokens(full_ids: list[int], input_len: int) -> int:
    return max(0, len(full_ids) - input_len)


def summarize_run(name, elapsed_s, total_input_tokens, total_output_tokens,
                  latencies_us=None):
    out = max(total_output_tokens, 1)
    result = {
        "name": name,
        "elapsed_s": elapsed_s,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "tpot_ms": (elapsed_s * 1000.0 / out),
        "throughput": (total_output_tokens / elapsed_s) if elapsed_s > 0 else 0.0,
    }
    if latencies_us:
        sorted_lat = sorted(latencies_us)
        n = len(sorted_lat)
        result["lat_avg"] = sum(latencies_us) / n
        result["lat_p50"] = sorted_lat[n // 2]
        result["lat_p90"] = sorted_lat[max(int(n * 0.9) - 1, 0)]
    return result


def print_speedup(label: str, baseline: dict, candidate: dict):
    base = baseline["throughput"]
    cand = candidate["throughput"]
    ratio = cand / base if base > 0 else 0.0
    print(f"  Throughput speedup ({label}): {ratio:.2f}x "
          f"[{candidate['name']} {cand:.1f} t/s ÷ {baseline['name']} {base:.1f} t/s]")


def format_single_table(rows: list[dict], hf_stats: dict, ll_stats: dict) -> str:
    hdr = (f"  {'case':<10s} {'in':>6s} {'ll_out':>8s} {'hf_out':>8s} "
           f"{'ll_ms':>10s} {'hf_ms':>10s} {'speedup':>8s}")
    W = len(hdr)
    lines = [f"\n  {'═' * W}", hdr, f"  {'─' * W}"]
    for r in rows:
        ll_ms = r["ll_us"] / 1000.0
        hf_ms = r["hf_us"] / 1000.0
        ll_tp = (r["ll_out"] / ll_ms * 1000) if ll_ms > 0 and r["ll_out"] else 0
        hf_tp = (r["hf_out"] / hf_ms * 1000) if hf_ms > 0 and r["hf_out"] else 0
        sp = ll_tp / hf_tp if hf_tp > 0 else 0
        lines.append(
            f"  {r['label']:<10s} {r['input_len']:>6d} {r['ll_out']:>8d} {r['hf_out']:>8d} "
            f"{ll_ms:>10.1f} {hf_ms:>10.1f} {sp:>7.2f}x"
        )
    lines.append(f"  {'═' * W}")
    lines.append(f"\n  {'Metric':<16s} {'LLAISYS':>14s} {'HF':>14s}")
    lines.append(f"  {'─' * 46}")
    lines.append(
        f"  {'wall':<16s} {ll_stats['elapsed_s']*1000:11.1f} ms "
        f"{hf_stats['elapsed_s']*1000:11.1f} ms"
    )
    lines.append(
        f"  {'out tokens':<16s} {ll_stats['output_tokens']:14d} "
        f"{hf_stats['output_tokens']:14d}"
    )
    lines.append(
        f"  {'TPOT':<16s} {ll_stats['tpot_ms']:11.2f} ms {hf_stats['tpot_ms']:11.2f} ms"
    )
    lines.append(
        f"  {'Throughput':<16s} {ll_stats['throughput']:11.1f} t/s "
        f"{hf_stats['throughput']:11.1f} t/s"
    )
    lines.append(
        f"  {'TTFT(avg)':<16s} {ll_stats.get('ttft_avg_ms', 0):11.1f} ms "
        f"{hf_stats.get('ttft_avg_ms', 0):11.1f} ms"
    )
    return "\n".join(lines)


def save_json(payload: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{ts}")
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, "results.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"  JSON saved: {path}")
    return path


def release_hf(_model=None):
    """Flush Torch CUDA cache after HF weights are unreachable.

    Must be called **after** the caller drops its last reference::

        hf_model = None
        release_hf()
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        torch.cuda.synchronize()
