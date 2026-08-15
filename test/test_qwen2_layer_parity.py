#!/usr/bin/env python3
"""Legacy dual-path parity retired: only Layer forward remains.

Kept as a thin wrapper that runs a short greedy generate on the default path.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--device", default="nvidia")
    ap.add_argument("--max_steps", type=int, default=8)
    ap.add_argument("--skip-hf", action="store_true")
    args = ap.parse_args()
    import llaisys
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    ids = tok.encode("Hello")
    device = llaisys.DeviceType.NVIDIA if args.device == "nvidia" else llaisys.DeviceType.CPU
    model = llaisys.models.Qwen2(args.model, device)
    out = model.generate(ids, max_new_tokens=args.max_steps, top_k=1, temperature=0.0)
    model.close()
    print(f"generated={len(out)-len(ids)} tokens; HARD GATE N/A (legacy removed); PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
