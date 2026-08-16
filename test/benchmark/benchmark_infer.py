#!/usr/bin/env python3
"""Single-request serial benchmark — HF generate vs LLAISYS generate.

Inputs are constructed to exact token lengths and shared across backends.
On small GPUs use ``--backend hf`` then ``--backend llaisys --from-json …``
so the two weight sets never share one process.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from infer_utils import (
    build_workload,
    count_new_tokens,
    format_single_table,
    hf_generate_serial,
    llaisys_generate_serial,
    load_hf_model,
    load_llaisys_model,
    make_warmup_ids,
    parse_int_list,
    print_speedup,
    release_hf,
    save_json,
    summarize_run,
)


def _load_tokenizer_only(model_path: str):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_steps", default=64, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument(
        "--input-lens",
        default="32,128,512",
        help="Comma-separated exact input token lengths",
    )
    parser.add_argument("--prompts-per-len", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument(
        "--force-max-tokens",
        action="store_true",
        help="Ask HF for min_new_tokens=max_new_tokens (LLAISYS best-effort)",
    )
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default="test/benchmark/results")
    parser.add_argument(
        "--backend",
        choices=["both", "hf", "llaisys"],
        default="both",
        help="Run one or both backends (use split processes on 8GB GPUs)",
    )
    parser.add_argument(
        "--from-json",
        default=None,
        help="Partial results JSON from --backend hf (for --backend llaisys)",
    )
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0
        args.top_k = 1
        args.temperature = 0.0

    torch.manual_seed(args.seed)
    input_lens = parse_int_list(args.input_lens)
    max_steps = args.max_steps
    tp, tk, temp = args.top_p, args.top_k, args.temperature
    force = args.force_max_tokens

    partial = None
    if args.from_json:
        with open(args.from_json) as f:
            partial = json.load(f)

    cases = None
    model_path = args.model
    hf_stats = hf_best = rows_hf = None
    vocab = None

    if args.backend in ("both", "hf"):
        tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)
        cases = build_workload(
            tokenizer,
            input_lens,
            args.prompts_per_len,
            use_chat_template=not args.no_chat_template,
        )
        vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer)

        for _ in range(args.warmup):
            warm_cases = [
                {**c, "input_ids": make_warmup_ids(c["input_ids"], vocab)} for c in cases
            ]
            hf_generate_serial(
                warm_cases, tokenizer, hf_model, max_steps, tp, tk, temp,
                force_max_tokens=force,
            )

        hf_best = None
        hf_wall_best = None
        for _ in range(args.repeat):
            hf_results, wall_s = hf_generate_serial(
                cases, tokenizer, hf_model, max_steps, tp, tk, temp,
                force_max_tokens=force,
            )
            if hf_wall_best is None or wall_s < hf_wall_best:
                hf_wall_best = wall_s
                hf_best = hf_results

        hf_out = sum(
            count_new_tokens(full, c["input_len"]) for c, (full, _) in zip(cases, hf_best)
        )
        hf_in = sum(c["input_len"] for c in cases)
        hf_lat = [e_us for _, e_us in hf_best]
        hf_stats = summarize_run("HF", hf_wall_best, hf_in, hf_out, hf_lat)
        hf_stats["ttft_avg_ms"] = 0.0
        hf_model = None
        release_hf()

        if args.backend == "hf":
            payload = {
                "environment": {
                    "gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda,
                },
                "config": {
                    "mode": "single-serial",
                    "input_lens": input_lens,
                    "prompts_per_len": args.prompts_per_len,
                    "max_steps": max_steps,
                    "force_max_tokens": force,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "seed": args.seed,
                    "model_path": model_path,
                    "no_chat_template": args.no_chat_template,
                },
                "cases": cases,
                "results": {
                    "hf": hf_stats,
                    "hf_full_ids": [full for full, _ in hf_best],
                    "hf_lat_us": [e_us for _, e_us in hf_best],
                },
            }
            out_path = save_json(payload, args.output)
            print(f"  HF-only partial saved: {out_path}")
            return

    if args.backend == "llaisys":
        if partial is None:
            parser.error("--backend llaisys requires --from-json")
        model_path = args.model or partial["config"]["model_path"]
        cases = partial["cases"]
        hf_stats = partial["results"]["hf"]
        hf_best = list(
            zip(partial["results"]["hf_full_ids"], partial["results"]["hf_lat_us"])
        )
        tokenizer = _load_tokenizer_only(model_path)
        vocab = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
        input_lens = partial["config"]["input_lens"]
        max_steps = partial["config"]["max_steps"]
        force = partial["config"]["force_max_tokens"]
        args.warmup = partial["config"].get("warmup", args.warmup)
        args.repeat = partial["config"].get("repeat", args.repeat)
        args.prompts_per_len = partial["config"].get(
            "prompts_per_len", args.prompts_per_len
        )

    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)
    for _ in range(args.warmup):
        warm_cases = [
            {**c, "input_ids": make_warmup_ids(c["input_ids"], vocab)} for c in cases
        ]
        llaisys_generate_serial(warm_cases, ll_model, max_steps, tp, tk, temp)

    ll_best = None
    ll_wall_best = None
    for _ in range(args.repeat):
        ll_results, wall_s = llaisys_generate_serial(
            cases, ll_model, max_steps, tp, tk, temp,
        )
        if ll_wall_best is None or wall_s < ll_wall_best:
            ll_wall_best = wall_s
            ll_best = ll_results

    hf_in = sum(c["input_len"] for c in cases)
    ll_out = sum(
        count_new_tokens(full, c["input_len"]) for c, (full, _, _) in zip(cases, ll_best)
    )
    ll_lat = [e_us for _, e_us, _ in ll_best]
    ll_ttft = [ttft for _, _, ttft in ll_best]
    ll_stats = summarize_run("LLAISYS", ll_wall_best, hf_in, ll_out, ll_lat)
    ll_stats["ttft_avg_ms"] = (sum(ll_ttft) / len(ll_ttft) / 1000.0) if ll_ttft else 0.0

    rows = []
    for i, c in enumerate(cases):
        rows.append({
            "label": c["label"],
            "input_len": c["input_len"],
            "ll_out": count_new_tokens(ll_best[i][0], c["input_len"]),
            "hf_out": count_new_tokens(hf_best[i][0], c["input_len"]),
            "ll_us": ll_best[i][1],
            "hf_us": hf_best[i][1],
        })

    gpu = torch.cuda.get_device_name(0)
    print(
        f"\n  Model: {Path(model_path).name}  |  GPU: {gpu}  |  Mode: single-serial\n"
        f"  input_lens={input_lens}  prompts_per_len={args.prompts_per_len}  "
        f"max_steps={max_steps}  force_max_tokens={force}  "
        f"warmup={args.warmup} repeat={args.repeat} (best wall)"
    )
    print(format_single_table(rows, hf_stats, ll_stats))
    print_speedup("vs HF generate serial", hf_stats, ll_stats)

    if args.test:
        matched = 0
        for idx, (hf_r, ll_r) in enumerate(zip(hf_best, ll_best)):
            ok = hf_r[0] == ll_r[0]
            matched += int(ok)
            print(
                f"  Case [{cases[idx]['label']}] {'OK' if ok else 'MISMATCH'}  "
                f"hf={len(hf_r[0])} ll={len(ll_r[0])}"
            )
        print(f"  Correctness: {matched}/{len(cases)} matched")
        if matched != len(cases):
            sys.exit(1)

    save_json(
        {
            "environment": {"gpu": gpu, "cuda": torch.version.cuda},
            "config": {
                "mode": "single-serial",
                "input_lens": input_lens,
                "prompts_per_len": args.prompts_per_len,
                "max_steps": max_steps,
                "force_max_tokens": force,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "seed": args.seed,
                "backend": args.backend,
            },
            "results": {"hf": hf_stats, "llaisys": ll_stats, "rows": rows},
            "speedup_vs_hf_serial": (
                ll_stats["throughput"] / hf_stats["throughput"]
                if hf_stats["throughput"] else 0
            ),
        },
        args.output,
    )
    ll_model.close()


if __name__ == "__main__":
    main()
