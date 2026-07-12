#!/usr/bin/env python3
"""Run inference to trigger profiling. Profiler auto-dumps at model stop."""

import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
import llaisys
from transformers import AutoTokenizer

PROMPTS = [
    "What is the capital of France?",
    "Explain how backpropagation works in neural networks.",
    "You are designing a GPU inference system. Describe the key components.",
    "Write a Python function to compute the Fibonacci sequence efficiently.",
    "Summarize the transformer architecture in three paragraphs.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--max_steps", default=1025, type=int)
    parser.add_argument("--prompt-index", default=-1, type=int,
                        help="Run single prompt by index (0-4), default runs all")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = llaisys.models.Qwen2(args.model, llaisys.DeviceType.NVIDIA)

    for i, prompt in enumerate(PROMPTS):
        if args.prompt_index >= 0 and i != args.prompt_index:
            continue
        print(f"  Prompt {i+1}/{len(PROMPTS)}: {prompt[:60]}...")
        input_ids = tokenizer.encode(prompt)
        tokens = model.generate(input_ids, max_new_tokens=args.max_steps)
        print(f"    Generated {len(tokens) - len(input_ids)} tokens")

    # Explicit close → C++ destructor → stop() → dump_json()
    model.close()
    print("Profile auto-saved to profile_qwen2.json")

if __name__ == "__main__":
    main()
