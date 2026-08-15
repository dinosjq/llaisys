#!/usr/bin/env python3
"""Wall-clock P0/D0/D1 gate: legacy vs layer (≤3% regression)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer  # noqa: E402

import llaisys  # noqa: E402
from test.profile.nsys_profile import (  # noqa: E402
    _PROMPT_SEEDS,
    _build_prompt,
    make_warmup_input,
)


SHAPES = {
    "P0": {"input_len": 256, "max_steps": 1, "warmup_steps": 2},
    "D0": {"input_len": 1, "max_steps": 128, "warmup_steps": 2},
    "D1": {"input_len": 1024, "max_steps": 128, "warmup_steps": 2},
}


def build_input_ids(tokenizer, input_len: int) -> list[int]:
    seed = _build_prompt(input_len * 4, "en")
    content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": seed}],
        add_generation_prompt=True,
        tokenize=False,
    )
    input_ids = tokenizer.encode(content)
    if len(input_ids) > input_len:
        return input_ids[:input_len]
    while len(input_ids) < input_len:
        extra = tokenizer.encode(" " + _build_prompt(input_len, "en"))
        input_ids = (input_ids + extra)[:input_len]
    return input_ids


def run_once(model_path: str, mode: str, shape: str, input_ids: list[int], vocab_size: int) -> dict:
    env = os.environ.copy()
    env["LLAISYS_QWEN2_LAYER_FORWARD"] = "0" if mode == "legacy" else "1"
    # Subprocess isolates CUDA teardown SEGV between modes.
    payload = {
        "model": model_path,
        "input_ids": input_ids,
        "vocab_size": vocab_size,
        **SHAPES[shape],
    }
    code = f"""
import json, os, sys, time
sys.path.insert(0, {str(ROOT)!r})
import llaisys
from test.profile.nsys_profile import make_warmup_input
cfg = json.loads({json.dumps(payload)!r})
model = llaisys.models.Qwen2(cfg['model'], llaisys.DeviceType.NVIDIA)
ids = cfg['input_ids']
if cfg['warmup_steps']:
    warm = make_warmup_input(ids, cfg['vocab_size'])
    _ = model.generate(warm, max_new_tokens=cfg['warmup_steps'])
t0 = time.perf_counter()
out = model.generate(ids, max_new_tokens=cfg['max_steps'])
elapsed = time.perf_counter() - t0
n_out = len(out) - len(ids)
model.close()
print(json.dumps({{'elapsed_s': elapsed, 'output_tokens': n_out, 'input_tokens': len(ids)}}))
"""
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{mode}/{shape} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout[-2000:]}\nstderr:\n{proc.stderr[-2000:]}"
        )
    # Last JSON line
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip().startswith("{")]
    if not lines:
        raise RuntimeError(f"no JSON from {mode}/{shape}:\n{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
    data = json.loads(lines[-1])
    out_tok = max(1, int(data["output_tokens"]))
    data["wall_per_output_token_ms"] = 1000.0 * float(data["elapsed_s"]) / out_tok
    data["mode"] = mode
    data["shape"] = shape
    return data


def median(xs: list[float]) -> float:
    return float(statistics.median(xs))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--shapes", default="P0,D0,D1")
    ap.add_argument("--threshold", type=float, default=0.03, help="max allowed relative regression")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    shapes = [s.strip() for s in args.shapes.split(",") if s.strip()]
    results = {"repeats": args.repeats, "threshold": args.threshold, "shapes": {}}

    for shape in shapes:
        input_ids = build_input_ids(tokenizer, SHAPES[shape]["input_len"])
        print(f"\n=== {shape} input_len={len(input_ids)} max_steps={SHAPES[shape]['max_steps']} ===", flush=True)
        legacy_ms, layer_ms = [], []
        for i in range(args.repeats):
            print(f"  [{i+1}/{args.repeats}] legacy...", flush=True)
            leg = run_once(args.model, "legacy", shape, input_ids, tokenizer.vocab_size)
            print(f"  [{i+1}/{args.repeats}] layer...", flush=True)
            lay = run_once(args.model, "layer", shape, input_ids, tokenizer.vocab_size)
            legacy_ms.append(leg["wall_per_output_token_ms"])
            layer_ms.append(lay["wall_per_output_token_ms"])
            print(
                f"    legacy={leg['wall_per_output_token_ms']:.3f} ms/tok  "
                f"layer={lay['wall_per_output_token_ms']:.3f} ms/tok  "
                f"elapsed L/R={lay['elapsed_s']:.3f}/{leg['elapsed_s']:.3f}s",
                flush=True,
            )
        med_l = median(legacy_ms)
        med_y = median(layer_ms)
        rel = (med_y - med_l) / med_l if med_l > 0 else float("inf")
        passed = rel <= args.threshold
        results["shapes"][shape] = {
            "legacy_ms_tok": legacy_ms,
            "layer_ms_tok": layer_ms,
            "median_legacy_ms_tok": med_l,
            "median_layer_ms_tok": med_y,
            "relative_regression": rel,
            "pass": passed,
        }
        print(
            f"  MEDIAN legacy={med_l:.3f} layer={med_y:.3f} "
            f"rel={rel*100:.2f}%  {'PASS' if passed else 'FAIL'}",
            flush=True,
        )

    overall = all(v["pass"] for v in results["shapes"].values())
    results["overall_pass"] = overall
    text = json.dumps(results, indent=2)
    print("\n" + text, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
        print(f"wrote {args.out}", flush=True)
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
