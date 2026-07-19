#!/usr/bin/env python3
"""Inference profiler — single-prompt operator timing across KV cache lengths."""

import argparse, json, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import llaisys
from transformers import AutoTokenizer

PROMPTS = {
    10:   "What is AI?",
    50:   "Explain how backpropagation works in neural networks. "
          "Include the chain rule and gradient flow through hidden layers.",
    100:  "Describe the transformer architecture in detail. "
          "Cover self-attention, multi-head attention, position encoding, "
          "and the encoder-decoder structure. Explain why transformers "
          "replaced RNNs for sequence modeling tasks.",
    500:  "You are designing a GPU inference serving system for large "
          "language models. The system must handle concurrent requests "
          "with varying prompt lengths and generate responses up to 2048 "
          "tokens each. Analyze memory planning for KV cache blocks, "
          "batching strategy including continuous batching and chunked "
          "prefill, scheduling policy for throughput vs latency SLOs, "
          "and quantization tradeoffs for FP8 or INT8 KV cache. "
          "Provide concrete numbers and reasoning for each design choice.",
}

OUTPUT_DIR = "test/profile/results"


def format_table(profile: dict, input_tok: int, output_tok: int,
                 elapsed_s: float) -> str:
    lines = []
    prefill_ms = profile["snapshots"][0]["total_ms"]
    throughput = output_tok / elapsed_s if elapsed_s > 0 else 0
    tpot_ms = elapsed_s * 1000 / output_tok if output_tok > 0 else 0

    lines.append(
        f"  input_token: {input_tok}   output_token: {output_tok}   "
        f"elapsed: {elapsed_s:.2f}s"
    )
    lines.append(
        f"  throughput: {throughput:.1f} tok/s   "
        f"TTFT: {prefill_ms:.2f} ms   TPOT: {tpot_ms:.2f} ms"
    )

    for snap in profile["snapshots"]:
        phase, step, total, ops = (
            snap["phase"], snap["step"], snap["total_ms"], snap["ops"])
        lines.append(
            f"\n  period: {phase}   step: {step}   total_ms: {total:.2f}")
        lines.append(
            f"    {'operator':<22s} {'avg_ms':>8s}  {'count':>5s}  "
            f"{'total_ms':>9s}  {'%':>6s}")
        lines.append(f"    {'─' * 57}")

        for op in ops:
            pct = (op["total_ms"] / total * 100) if total > 0 else 0
            lines.append(
                f"    {op['name']:<22s} {op['avg_ms']:8.2f}  "
                f"{op['count']:5d}  {op['total_ms']:9.2f}  {pct:5.1f}")
        lines.append(f"    {'─' * 57}")
        lines.append(
            f"    {'total':<22s} {'':>8s}  {'':>5s}  {total:9.2f}  {100.0:5.1f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", default=1025, type=int)
    parser.add_argument("--prompt-len", default=50, type=int,
                        help=f"Target input token length. "
                             f"Available: {sorted(PROMPTS.keys())}")
    parser.add_argument("--prompt", default="", type=str,
                        help="Custom prompt (overrides --prompt-len)")
    args = parser.parse_args()

    prompt = args.prompt if args.prompt else PROMPTS.get(
        args.prompt_len, PROMPTS[50])

    tokenizer = AutoTokenizer.from_pretrained(args.model,
                                              trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)

    input_ids = tokenizer.encode(prompt)
    input_tok = len(input_ids)

    t0 = time.perf_counter()
    tokens = model.generate(input_ids, max_new_tokens=args.max_steps)
    elapsed_s = time.perf_counter() - t0
    output_tok = len(tokens) - input_tok

    model.close()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(OUTPUT_DIR, f"profile_{ts}.json")

    # Move the file if profiler wrote to a fixed path
    default_path = os.path.join(OUTPUT_DIR, "profile_qwen2.json")
    if os.path.exists(default_path):
        os.rename(default_path, json_path)

    with open(json_path) as f:
        profile = json.load(f)

    print(format_table(profile, input_tok, output_tok, elapsed_s))


if __name__ == "__main__":
    main()
