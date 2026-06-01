import argparse
import gc
import os
import threading
import time
import warnings

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging

import llaisys


hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")


def torch_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    if device_name == "nvidia":
        return torch.device("cuda:0")
    raise ValueError(f"Unsupported device name: {device_name}")


def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    if device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    raise ValueError(f"Unsupported device name: {device_name}")


def sync_device(device_name: str):
    if device_name == "nvidia" and torch.cuda.is_available():
        torch.cuda.synchronize()


def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = model.to(torch_device(device_name))
    # 添加专用 pad_token（不与 eos_token 冲突，否则 batch generate 不会正常停）
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model, model_path


def load_llaisys_model(model_path, device_name):
    return llaisys.models.Qwen2(model_path, llaisys_device(device_name))


def build_input_tokens(tokenizer, prompt):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(input_content)


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
        context_lines = []
        for repeat_idx in range(context_repeat):
            context_lines.append(
                f"Context line {repeat_idx + 1}: this benchmark should stress input handling, decoding, and batching behavior."
            )
        prompt = topic + "\n\n" + extra
        if context_lines:
            prompt += "\n\n" + "\n".join(context_lines)
        prompts.append(prompt)
    return prompts


def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8):
    input_ids = build_input_tokens(tokenizer, prompt)
    start = time.perf_counter()
    result_tokens = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )
    elapsed = time.perf_counter() - start
    return input_ids, result_tokens, tokenizer.decode(result_tokens, skip_special_tokens=True), elapsed


def run_single_ll(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature, results, idx):
    results[idx] = llaisys_infer(
        prompt,
        tokenizer,
        model,
        max_new_tokens=max_new_tokens,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )


def run_hf_batch(prompts, tokenizer, model, max_new_tokens, top_p, top_k, temperature, device_name):
    # Tokenize all prompts with padding for true batch inference
    input_contents = [
        tokenizer.apply_chat_template(
            conversation=[{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        for prompt in prompts
    ]
    enc = tokenizer(
        input_contents,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to(model.device)

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_cache=True,
        )
    sync_device(device_name)
    elapsed = time.perf_counter() - start

    input_lens = enc.attention_mask.sum(dim=1).tolist()
    max_input_len = enc.input_ids.shape[1]
    results = []
    for idx in range(len(prompts)):
        input_ids = outputs[idx, :input_lens[idx]].tolist()
        # Strip input padding: take actual input tokens + generated tokens (after max_input_len)
        full = outputs[idx].tolist()
        generated_part = full[max_input_len:]
        # 去掉尾部 pad（提前结束的序列会被 HF 用 pad_token 填满）
        while generated_part and generated_part[-1] == tokenizer.pad_token_id:
            generated_part.pop()
        output_ids = input_ids + generated_part
        olen = len(generated_part)
        print(f"  HF [{idx}] input={input_lens[idx]} output={olen} total={len(output_ids)} batch_time={elapsed*1000.0:.3f} ms")
        results.append((
            input_ids,
            output_ids,
            tokenizer.decode(output_ids, skip_special_tokens=True),
            elapsed,
        ))
    return results, elapsed


def run_ll_batch(prompts, tokenizer, model, max_new_tokens, top_p, top_k, temperature, device_name):
    results = [None] * len(prompts)
    threads = []
    start = time.perf_counter()
    for idx, prompt in enumerate(prompts):
        thread = threading.Thread(
            target=run_single_ll,
            args=(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature, results, idx),
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    sync_device(device_name)
    elapsed = time.perf_counter() - start
    return results, elapsed


def summarize_run(name, elapsed_s, results):
    input_tokens = sum(len(item[0]) for item in results)
    output_tokens = sum(len(item[1]) - len(item[0]) for item in results)
    total_tokens = sum(len(item[1]) for item in results)
    prompt_latencies = [item[3] for item in results]
    tpot_ms = (elapsed_s * 1000.0 / output_tokens) if output_tokens else 0.0
    throughput = (output_tokens / elapsed_s) if elapsed_s > 0 else 0.0
    avg_prompt_latency = elapsed_s / len(results) if results else 0.0
    avg_input_tokens = (input_tokens / len(results)) if results else 0.0
    avg_output_tokens = (output_tokens / len(results)) if results else 0.0
    p50_latency = sorted(prompt_latencies)[len(prompt_latencies) // 2] if prompt_latencies else 0.0
    p90_latency = sorted(prompt_latencies)[max(int(len(prompt_latencies) * 0.9) - 1, 0)] if prompt_latencies else 0.0
    max_latency = max(prompt_latencies) if prompt_latencies else 0.0

    print(f"\n=== {name} ===")
    print(f"Total elapsed: {elapsed_s:.3f}s")
    print(f"Input tokens:  {input_tokens}")
    print(f"Output tokens: {output_tokens}")
    print(f"Total tokens:  {total_tokens}")
    print(f"Avg input tokens / prompt:  {avg_input_tokens:.2f}")
    print(f"Avg output tokens / prompt: {avg_output_tokens:.2f}")
    print(f"Avg latency / prompt: {avg_prompt_latency * 1000.0:.3f} ms")
    print(f"Prompt latency p50/p90/max: {p50_latency*1000.0:.3f} / {p90_latency*1000.0:.3f} / {max_latency*1000.0:.3f} ms")
    print(f"TPOT: {tpot_ms:.4f} ms/token")
    print(f"Throughput: {throughput:.3f} output tok/s")
    return {
        "elapsed_s": elapsed_s,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "tpot_ms": tpot_ms,
        "throughput": throughput,
    }


def print_speedup(hf_stats, ll_stats):
    if ll_stats["elapsed_s"] <= 0:
        return
    speedup = hf_stats["elapsed_s"] / ll_stats["elapsed_s"]
    tpot_speedup = hf_stats["tpot_ms"] / ll_stats["tpot_ms"] if ll_stats["tpot_ms"] > 0 else 0.0
    print("\n=== Speedup ===")
    print(f"Wall-clock speedup vs HF: {speedup:.2f}x")
    print(f"TPOT improvement vs HF:   {tpot_speedup:.2f}x")
    print(f"Latency reduction:        {(1.0 - ll_stats['elapsed_s'] / hf_stats['elapsed_s']) * 100.0:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_prompts", default=12, type=int)
    parser.add_argument("--context_repeat", default=2, type=int)
    parser.add_argument("--warmup", default=0, type=int)
    parser.add_argument("--repeat", default=1, type=int)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--show_mismatch", action="store_true")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0
        args.top_k = 1
        args.temperature = 1.0
        args.warmup = 0
        args.repeat = 1
        args.show_mismatch = False

    prompts = build_prompt_bank(args.num_prompts, args.context_repeat)

    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)

    max_new_tokens = args.max_steps
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature

    print(f"Running performance test on {len(prompts)} prompts")
    print(f"Prompt variants: {args.num_prompts}, context_repeat: {args.context_repeat}")
    print(f"Max new tokens: {max_new_tokens}")
    print(f"Sampling: top_p={top_p}, top_k={top_k}, temperature={temperature}")

    for _ in range(args.warmup):
        run_hf_batch(prompts, tokenizer, hf_model, max_new_tokens, top_p, top_k, temperature, args.device)

    hf_last_results = None
    hf_elapsed_sum = 0.0
    for _ in range(args.repeat):
        hf_last_results, hf_elapsed = run_hf_batch(
            prompts, tokenizer, hf_model, max_new_tokens, top_p, top_k, temperature, args.device
        )
        hf_elapsed_sum += hf_elapsed

    hf_stats = summarize_run("Baseline (HF)", hf_elapsed_sum / args.repeat, hf_last_results)

    del hf_model
    gc.collect()
    if args.device == "nvidia" and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print("\nLoading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)

    for _ in range(args.warmup):
        run_ll_batch(prompts, tokenizer, ll_model, max_new_tokens, top_p, top_k, temperature, args.device)

    ll_last_results = None
    ll_elapsed_sum = 0.0
    for _ in range(args.repeat):
        ll_last_results, ll_elapsed = run_ll_batch(
            prompts, tokenizer, ll_model, max_new_tokens, top_p, top_k, temperature, args.device
        )
        ll_elapsed_sum += ll_elapsed

    ll_stats = summarize_run("Your Result (LLAISYS)", ll_elapsed_sum / args.repeat, ll_last_results)

    print("\n=== Correctness ===")
    all_ok = True
    matched = 0
    for idx, (hf_item, ll_item) in enumerate(zip(hf_last_results, ll_last_results)):
        hf_tokens = hf_item[1]
        ll_tokens = ll_item[1]
        ok = hf_tokens == ll_tokens
        all_ok = all_ok and ok
        matched += int(ok)
        print(f"  Prompt [{idx}] ok={ok} hf_len={len(hf_tokens)} ll_len={len(ll_tokens)}")
        if not ok and args.show_mismatch:
            print("    HF text:\n", hf_item[2])
            print("    LL text:\n", ll_item[2])

    print_speedup(hf_stats, ll_stats)

    print(f"Matched prompts: {matched}/{len(prompts)}")
    if args.strict and not all_ok:
        raise SystemExit("Performance test failed: outputs differ")

    if all_ok:
        print("\n\033[92mPerformance test passed!\033[0m\n")
    else:
        print("\n\033[93mPerformance benchmark completed with output mismatches.\033[0m\n")