#!/usr/bin/env python3
"""Single-request serial inference benchmark — HF vs LLAISYS."""

import argparse, gc, sys, time
import torch
from infer_utils import (
    load_hf_model, load_llaisys_model, build_prompt_bank,
    hf_infer, llaisys_infer, summarize_run, print_speedup,
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

    # ── HF ───────────────────────────────────────────
    from infer_utils import _apply_chat
    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)
    prompt_input_lens = [len(_apply_chat(tokenizer, p)) for p in prompts]

    for _ in range(args.warmup):
        for p in prompts:
            hf_infer(p, tokenizer, hf_model, max_steps, tp, tk, temp)

    hf_results = []
    hf_ttfts = []
    hf_elapsed_sum = 0.0
    for _ in range(args.repeat):
        rep_results, rep_ttfts = [], []
        rep_start = time.perf_counter()
        for i, p in enumerate(prompts):
            tokens, e_us, ttft_us = hf_infer(p, tokenizer, hf_model, max_steps, tp, tk, temp)
            rep_results.append((tokens, e_us))
            rep_ttfts.append(ttft_us)
        hf_elapsed_sum += time.perf_counter() - rep_start
        hf_results, hf_ttfts = rep_results, rep_ttfts

    hf_in_tok  = sum(prompt_input_lens)
    hf_out_tok = sum(len(r[0]) - prompt_input_lens[i] for i, r in enumerate(hf_results))
    hf_lat_us  = [r[1] for r in hf_results]
    hf_stats = summarize_run("HF", hf_elapsed_sum / args.repeat, len(prompts),
                              hf_in_tok, hf_out_tok, hf_lat_us)
    hf_stats["per_prompt"] = hf_lat_us
    hf_stats["ttft_avg_ms"] = (sum(hf_ttfts) / len(hf_ttfts) / 1000.0) if hf_ttfts else 0
    del hf_model; gc.collect()
    torch.cuda.empty_cache(); torch.cuda.synchronize()

    # ── LLAISYS ──────────────────────────────────────
    WARMUP_PROMPT = "Hello, this is a dedicated warmup prompt for GPU initialization."
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)
    for _ in range(args.warmup):
        llaisys_infer(WARMUP_PROMPT, tokenizer, ll_model, max_steps, tp, tk, temp)

    ll_results = []
    ll_elapsed_sum = 0.0
    for _ in range(args.repeat):
        rep_results = []
        rep_start = time.perf_counter()
        for i, p in enumerate(prompts):
            tokens, e_us, _ = llaisys_infer(p, tokenizer, ll_model, max_steps, tp, tk, temp)
            rep_results.append((tokens, e_us))
        ll_elapsed_sum += time.perf_counter() - rep_start
        ll_results = rep_results

    ll_out_tok = sum(len(r[0]) - prompt_input_lens[i] for i, r in enumerate(ll_results))
    ll_lat_us  = [r[1] for r in ll_results]
    ll_stats = summarize_run("LLAISYS", ll_elapsed_sum / args.repeat, len(prompts),
                              hf_in_tok, ll_out_tok, ll_lat_us)

    # ── Per-prompt results list ──────────────────────
    results_list = []
    tiers = ["short", "medium", "long"]
    for i, p in enumerate(prompts):
        tier = tiers[i % len(tiers)]
        results_list.append({
            "label": f"{tier}_{i//len(tiers)}",
            "input_tok": prompt_input_lens[i],
            "ll_out": len(ll_results[i][0]) - prompt_input_lens[i],
            "hf_out": len(hf_results[i][0]) - prompt_input_lens[i],
            "ll_ms": ll_lat_us[i],
        })

    # ── Output ───────────────────────────────────────
    gpu = torch.cuda.get_device_name(0)
    mode = "single"
    print(f"\n  Model: Qwen2-1.5B  |  GPU: {gpu}  |  Prompts: {args.num_prompts}  |  Mode: {mode}")
    print(format_table(results_list, hf_stats, ll_stats, mode))
    print_speedup(hf_stats, ll_stats)

    if args.test:
        matched = 0
        for idx, (hf_r, ll_r) in enumerate(zip(hf_results, ll_results)):
            ok = hf_r[0] == ll_r[0]
            matched += int(ok)
            print(f"  Prompt [{idx}] {'OK' if ok else 'MISMATCH'}  hf={len(hf_r[0])}  ll={len(ll_r[0])}")
        print(f"  Correctness: {matched}/{len(prompts)} matched")
        if matched != len(prompts):
            sys.exit(1)

    save_json(hf_stats, ll_stats,
              {"mode": mode, "num_prompts": args.num_prompts,
               "max_steps": max_steps, "seed": args.seed},
              args.output)


if __name__ == "__main__":
    main()
