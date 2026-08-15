#!/usr/bin/env python3
"""Short Llama-3.2 vertical-slice smoke: load + greedy generate a few tokens."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from transformers import AutoTokenizer

import llaisys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/songjq/models/Llama-3.2-3B-Instruct")
    ap.add_argument("--device", choices=["cpu", "nvidia"], default="nvidia")
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--prompt", default="Hello")
    args = ap.parse_args()

    device = llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.CPU
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": args.prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(text)
    print(f"device={args.device} prompt_tokens={len(input_ids)} stack={os.environ.get('LLAISYS_LAYER_STACK')}")

    model = llaisys.models.Llama(args.model, device)
    print(f"after_ctor stack={os.environ.get('LLAISYS_LAYER_STACK')}")
    out = model.generate(input_ids, max_new_tokens=args.max_steps, top_k=1, temperature=0.0)
    model.close()

    new_ids = out[len(input_ids) :]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"generated_tokens={len(new_ids)} ids={new_ids}")
    print(f"decoded={decoded!r}")
    if len(new_ids) < 1:
        print("FAIL: no tokens generated")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
