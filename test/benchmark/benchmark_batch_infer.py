#!/usr/bin/env python3
"""Concurrent multi-request inference benchmark — HF batch vs LLAISYS concurrent."""

import argparse, gc, sys, time
import torch
from infer_utils import (
    load_hf_model, load_llaisys_model, build_prompt_bank,
    hf_batch_infer, llaisys_concurrent_infer, summarize_run, print_speedup,
    format_table, save_json,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", default=1024, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--num_prompts", default=12, type=int)
    parser.add_argument("--warmup", default=3, type=int)
    parser.add_argument("--repeat", default=5, type=int)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default="test/benchmark/results")
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0; args.top_k = 1; args.temperature = 1.0

    torch.manual_seed(args.seed)

    prompts = build_prompt_bank(args.num_prompts)
    max_steps, tp, tk, temp = args.max_steps, args.top_p, args.top_k, args.temperature

    # ── HF batch ─────────────────────────────────────
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)

    for _ in range(args.warmup):
        hf_batch_infer(prompts, tokenizer, hf_model, max_steps, tp, tk, temp)

    hf_elapsed_sum = 0.0
    hf_last = None
    for _ in range(args.repeat):
        hf_last, e_us = hf_batch_infer(prompts, tokenizer, hf_model, max_steps, tp, tk, temp)
        hf_elapsed_sum += e_us / 1e6  # us → s

    hf_in_tok  = sum(len(r[0]) for r in hf_last)
    hf_out_tok = sum(len(r[1]) - len(r[0]) for r in hf_last)
    hf_stats = summarize_run("HF", hf_elapsed_sum / args.repeat, len(prompts),
                              hf_in_tok, hf_out_tok, latencies_us=None)
    del hf_model; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ── LLAISYS concurrent ───────────────────────────
    WARMUP_PROMPT = "Hello, this is a dedicated warmup prompt for GPU initialization."
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)
    for _ in range(args.warmup):
        llaisys_concurrent_infer([WARMUP_PROMPT], tokenizer, ll_model, max_steps, tp, tk, temp)

    ll_elapsed_sum = 0.0
    ll_last = None
    for _ in range(args.repeat):
        ll_last, e_s = llaisys_concurrent_infer(prompts, tokenizer, ll_model, max_steps, tp, tk, temp)
        ll_elapsed_sum += e_s

    # llaisys_infer returns full output tokens (includes input), need prompt-level input lengths
    from infer_utils import _apply_chat
    ll_in_lens = [len(_apply_chat(tokenizer, p)) for p in prompts]
    ll_in_tok  = sum(ll_in_lens)
    ll_out_tok = sum(len(r[0]) - ll_in_lens[i] for i, r in enumerate(ll_last))
    ll_stats = summarize_run("LLAISYS", ll_elapsed_sum / args.repeat, len(prompts),
                              ll_in_tok, ll_out_tok, latencies_us=None)

    # ── Output ───────────────────────────────────────
    gpu = torch.cuda.get_device_name(0)
    results_list = [{
        "label": f"({args.num_prompts} prompts)",
        "input_tok": ll_in_tok,
        "ll_out": ll_out_tok,
        "hf_out": hf_out_tok,
        "ll_ms": (ll_elapsed_sum / args.repeat) * 1000.0,
    }]
    hf_stats["per_prompt"] = [(hf_elapsed_sum / args.repeat) * 1e6]
    hf_stats["ttft_avg_ms"] = 0
    print(f"\n  Model: Qwen2-1.5B  |  GPU: {gpu}  |  Prompts: {args.num_prompts}  |  Mode: concurrent")
    print(format_table(results_list, hf_stats, ll_stats, "concurrent"))
    print_speedup(hf_stats, ll_stats)

    if args.test:
        matched = 0
        for idx, (hf_r, ll_r) in enumerate(zip(hf_last, ll_last)):
            ok = hf_r[1] == ll_r[0]
            matched += int(ok)
            print(f"  Prompt [{idx}] {'OK' if ok else 'MISMATCH'}  hf={len(hf_r[1])}  ll={len(ll_r[0])}")
        print(f"  Correctness: {matched}/{len(prompts)} matched")
        if matched != len(prompts):
            sys.exit(1)

    save_json(hf_stats, ll_stats,
              {"mode": "concurrent", "num_prompts": args.num_prompts,
               "max_steps": max_steps, "seed": args.seed},
              args.output)


if __name__ == "__main__":
    main()
