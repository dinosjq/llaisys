#!/usr/bin/env python3
"""Concurrent benchmark — LLAISYS continuous batch vs HF serial (primary)
and HF padded batch (secondary).

Inputs are exact token lengths, shared across all backends.
On small GPUs: ``--backend hf`` then ``--backend llaisys --from-json …``.
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
    hf_generate_padded,
    hf_generate_serial,
    llaisys_generate_concurrent,
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
    parser.add_argument("--input-lens", default="32,128,512")
    parser.add_argument("--prompts-per-len", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--force-max-tokens", action="store_true")
    parser.add_argument("--no-chat-template", action="store_true")
    parser.add_argument("--skip-padded", action="store_true",
                        help="Skip secondary HF padded-batch baseline")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--output", default="test/benchmark/results")
    parser.add_argument(
        "--backend",
        choices=["both", "hf", "llaisys"],
        default="both",
        help="Run one or both backends (split processes on 8GB GPUs)",
    )
    parser.add_argument("--from-json", default=None,
                        help="Partial JSON from --backend hf")
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
    hf_serial_stats = hf_padded_stats = None
    hf_serial_best = None
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
        total_in = sum(c["input_len"] for c in cases)

        for _ in range(args.warmup):
            warm = [
                {**c, "input_ids": make_warmup_ids(c["input_ids"], vocab)} for c in cases
            ]
            hf_generate_serial(
                warm, tokenizer, hf_model, max_steps, tp, tk, temp,
                force_max_tokens=force,
            )

        hf_serial_best = None
        hf_serial_wall = None
        for _ in range(args.repeat):
            results, wall_s = hf_generate_serial(
                cases, tokenizer, hf_model, max_steps, tp, tk, temp,
                force_max_tokens=force,
            )
            if hf_serial_wall is None or wall_s < hf_serial_wall:
                hf_serial_wall = wall_s
                hf_serial_best = results

        hf_serial_out = sum(
            count_new_tokens(full, c["input_len"])
            for c, (full, _) in zip(cases, hf_serial_best)
        )
        hf_serial_stats = summarize_run(
            "HF-serial", hf_serial_wall, total_in, hf_serial_out
        )

        hf_padded_stats = None
        hf_padded_best = None
        if not args.skip_padded:
            for _ in range(max(1, args.warmup)):
                warm = [
                    {**c, "input_ids": make_warmup_ids(c["input_ids"], vocab)}
                    for c in cases
                ]
                hf_generate_padded(
                    warm, tokenizer, hf_model, max_steps, tp, tk, temp,
                    force_max_tokens=force,
                )
            best_us = None
            for _ in range(args.repeat):
                results, e_us = hf_generate_padded(
                    cases, tokenizer, hf_model, max_steps, tp, tk, temp,
                    force_max_tokens=force,
                )
                if best_us is None or e_us < best_us:
                    best_us = e_us
                    hf_padded_best = results
            padded_out = sum(
                count_new_tokens(full, c["input_len"])
                for c, (_, full) in zip(cases, hf_padded_best)
            )
            hf_padded_stats = summarize_run(
                "HF-padded", best_us / 1e6, total_in, padded_out
            )

        hf_model = None
        release_hf()

        if args.backend == "hf":
            payload = {
                "environment": {
                    "gpu": torch.cuda.get_device_name(0),
                    "cuda": torch.version.cuda,
                },
                "config": {
                    "mode": "concurrent",
                    "input_lens": input_lens,
                    "prompts_per_len": args.prompts_per_len,
                    "max_steps": max_steps,
                    "force_max_tokens": force,
                    "warmup": args.warmup,
                    "repeat": args.repeat,
                    "seed": args.seed,
                    "model_path": model_path,
                    "skip_padded": args.skip_padded,
                    "primary_baseline": "HF-serial",
                    "secondary_baseline": None if args.skip_padded else "HF-padded",
                },
                "cases": cases,
                "results": {
                    "hf_serial": hf_serial_stats,
                    "hf_padded": hf_padded_stats,
                    "hf_serial_full_ids": [full for full, _ in hf_serial_best],
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
        hf_serial_stats = partial["results"]["hf_serial"]
        hf_padded_stats = partial["results"].get("hf_padded")
        hf_serial_best = [
            (full, 0.0) for full in partial["results"]["hf_serial_full_ids"]
        ]
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

    total_in = sum(c["input_len"] for c in cases)
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)
    for _ in range(args.warmup):
        warm = [
            {**c, "input_ids": make_warmup_ids(c["input_ids"], vocab)} for c in cases
        ]
        llaisys_generate_concurrent(warm, ll_model, max_steps, tp, tk, temp)

    ll_best = None
    ll_wall = None
    for _ in range(args.repeat):
        results, wall_s = llaisys_generate_concurrent(
            cases, ll_model, max_steps, tp, tk, temp,
        )
        if ll_wall is None or wall_s < ll_wall:
            ll_wall = wall_s
            ll_best = results

    ll_out = sum(
        count_new_tokens(full, c["input_len"]) for c, (full, _, _) in zip(cases, ll_best)
    )
    ll_stats = summarize_run("LLAISYS-concurrent", ll_wall, total_in, ll_out)
    ttfts = [t for _, _, t in ll_best]
    ll_stats["ttft_avg_ms"] = (sum(ttfts) / len(ttfts) / 1000.0) if ttfts else 0.0

    gpu = torch.cuda.get_device_name(0)
    print(
        f"\n  Model: {Path(model_path).name}  |  GPU: {gpu}  |  Mode: concurrent\n"
        f"  input_lens={input_lens}  prompts_per_len={args.prompts_per_len}  "
        f"cases={len(cases)}  max_steps={max_steps}  force_max_tokens={force}\n"
        f"  warmup={args.warmup} repeat={args.repeat} (best wall)"
    )
    print(f"\n  {'Backend':<22s} {'wall_ms':>10s} {'out_tok':>8s} {'tok/s':>10s}")
    print(f"  {'─' * 54}")
    for st in (hf_serial_stats, hf_padded_stats, ll_stats):
        if st is None:
            continue
        print(
            f"  {st['name']:<22s} {st['elapsed_s']*1000:10.1f} "
            f"{st['output_tokens']:8d} {st['throughput']:10.1f}"
        )
    print()
    print_speedup("primary vs HF-serial", hf_serial_stats, ll_stats)
    if hf_padded_stats is not None:
        print_speedup("secondary vs HF-padded", hf_padded_stats, ll_stats)
        print_speedup("HF-padded vs HF-serial", hf_serial_stats, hf_padded_stats)

    if args.test:
        matched = 0
        for idx, ((hf_full, _), (ll_full, _, _)) in enumerate(
            zip(hf_serial_best, ll_best)
        ):
            ok = hf_full == ll_full
            matched += int(ok)
            print(
                f"  Case [{cases[idx]['label']}] {'OK' if ok else 'MISMATCH'}  "
                f"hf={len(hf_full)} ll={len(ll_full)}"
            )
        print(f"  Correctness: {matched}/{len(cases)} matched")
        if matched != len(cases):
            sys.exit(1)

    save_json(
        {
            "environment": {"gpu": gpu, "cuda": torch.version.cuda},
            "config": {
                "mode": "concurrent",
                "input_lens": input_lens,
                "prompts_per_len": args.prompts_per_len,
                "max_steps": max_steps,
                "force_max_tokens": force,
                "warmup": args.warmup,
                "repeat": args.repeat,
                "seed": args.seed,
                "primary_baseline": "HF-serial",
                "secondary_baseline": (
                    None if hf_padded_stats is None else "HF-padded"
                ),
                "backend": args.backend,
            },
            "results": {
                "hf_serial": hf_serial_stats,
                "hf_padded": hf_padded_stats,
                "llaisys": ll_stats,
            },
            "speedup_vs_hf_serial": (
                ll_stats["throughput"] / hf_serial_stats["throughput"]
                if hf_serial_stats["throughput"] else 0
            ),
            "speedup_vs_hf_padded": (
                (ll_stats["throughput"] / hf_padded_stats["throughput"])
                if hf_padded_stats and hf_padded_stats["throughput"] else None
            ),
        },
        args.output,
    )
    ll_model.close()


if __name__ == "__main__":
    main()
