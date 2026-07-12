#!/usr/bin/env python3
"""Inference profiler — samples operator timing across KV cache lengths."""

import argparse, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import llaisys
from transformers import AutoTokenizer

# Prompt tiers by target input token length
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
          "Provide concrete numbers and reasoning for each design choice. "
          "Consider the implications of different GPU architectures "
          "including memory bandwidth, compute throughput, and memory "
          "capacity constraints.",
    1000: "You are evaluating three GPU architectures for LLM inference "
          "deployment: NVIDIA A100 (80GB), H100 (80GB), and B200. Analyze "
          "each architecture's peak throughput under continuous batching, "
          "memory utilization breakdown including weights, KV cache, and "
          "activations, prefill vs decode bottlenecks at various sequence "
          "lengths, and cost per million tokens. Support your analysis with "
          "TFLOPS and memory bandwidth calculations. Then design an optimal "
          "serving configuration for a 7B-parameter model with FP16 weights "
          "and FP8 KV cache, serving a workload mix of 40% short prompts "
          "(<50 tokens) and 60% long prompts (500-2000 tokens) with average "
          "output lengths of 200 tokens. Which architecture would you choose "
          "and why? Consider both technical performance and total cost of "
          "ownership. Include edge cases like batch size limits, memory "
          "fragmentation, and the impact of different sequence length "
          "distributions on scheduler efficiency.",
}

OUTPUT_DIR = "test/profile/results"
JSON_PATH = os.path.join(OUTPUT_DIR, "profile_qwen2.json")


def format_table(profile: dict, input_tok: int, output_tok: int) -> str:
    """Render complete profiling table from JSON data."""
    lines = []
    ttft_ms = 0.0
    prefill_ms = 0.0

    for snap in profile["snapshots"]:
        phase = snap["phase"]
        step  = snap["step"]
        total = snap["total_ms"]
        ops   = snap["ops"]

        if phase == "prefill":
            prefill_ms = total
            ttft_ms = total

        # header
        lines.append(f"\n  period: {phase}   step: {step}   total_ms: {total:.2f}")
        lines.append(f"    {'operator':<22s} {'avg_ms':>8s}  {'count':>5s}  {'total_ms':>9s}  {'%':>6s}")
        lines.append(f"    {'─' * 57}")

        for op in ops:
            pct = (op["total_ms"] / total * 100.0) if total > 0 else 0.0
            lines.append(
                f"    {op['name']:<22s} {op['avg_ms']:8.2f}  {op['count']:5d}  "
                f"{op['total_ms']:9.2f}  {pct:5.1f}"
            )
        lines.append(f"    {'─' * 57}")
        lines.append(f"    {'total':<22s} {'':>8s}  {'':>5s}  {total:9.2f}  {100.0:5.1f}")

    # summary
    total_ms = prefill_ms + sum(
        s["total_ms"] for s in profile["snapshots"] if s["phase"] == "decode")
    throughput = (output_tok / total_ms * 1000.0) if total_ms > 0 else 0.0
    tpot = total_ms / output_tok if output_tok > 0 else 0.0

    lines.insert(0, (
        f"  input_token: {input_tok}   output_token: {output_tok}   total_ms: {total_ms:.2f}\n"
        f"  throughput: {throughput:.1f} tok/s   TTFT: {ttft_ms:.2f} ms   TPOT: {tpot:.2f} ms\n"
    ))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", default=1025, type=int)
    parser.add_argument("--prompt-len", default=0, type=int,
                        help=f"Target input length in tokens (0=all). "
                             f"Available: {sorted(PROMPTS.keys())}")
    args = parser.parse_args()

    lens_to_run = sorted(PROMPTS.keys()) if args.prompt_len == 0 else [args.prompt_len]

    for target_len in lens_to_run:
        prompt = PROMPTS[target_len]
        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
        model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)

        input_ids = tokenizer.encode(prompt)
        input_tok = len(input_ids)
        tokens = model.generate(input_ids, max_new_tokens=args.max_steps)
        output_tok = len(tokens) - input_tok
        model.close()

        with open(JSON_PATH) as f:
            profile = json.load(f)

        print(format_table(profile, input_tok, output_tok))


if __name__ == "__main__":
    main()
