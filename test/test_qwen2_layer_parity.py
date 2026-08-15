"""Hard gate: legacy vs layer-stack greedy token parity on a real Qwen2 model.

Fails if LLAISYS_QWEN2_LAYER_FORWARD=0 and =1 produce different token sequences
for the same prompt under greedy decoding (temperature<=0, top_k=1).

Default: two subprocesses (avoids CUDA teardown segfault when reloading a second
full model in-process). Optional --inprocess for single-process dual load.

HF comparison is diagnostic only (printed, never fails the gate).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Allow `python test/test_qwen2_layer_parity.py` from repo root / worktree.
_TEST_DIR = Path(__file__).resolve().parent
if str(_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(_TEST_DIR))

DEFAULT_MODEL = "/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B"
DEFAULT_PROMPT = "Who are you?"
ENV_FLAG = "LLAISYS_QWEN2_LAYER_FORWARD"


def _encode_prompt(tokenizer, prompt: str) -> list[int]:
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    return tokenizer.encode(input_content)


def _run_llaisys_greedy_inprocess(
    model_path: str,
    device_name: str,
    prompt_tokens: list[int],
    max_new_tokens: int,
    layer_forward: bool,
) -> list[int]:
    import torch
    import llaisys
    from test_utils import llaisys_device

    os.environ[ENV_FLAG] = "1" if layer_forward else "0"
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    try:
        outputs = model.generate(
            prompt_tokens,
            max_new_tokens=max_new_tokens,
            top_k=1,
            top_p=1.0,
            temperature=0.0,
        )
    finally:
        if hasattr(model, "close"):
            model.close()
        del model
        gc.collect()
        if device_name == "nvidia" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    return list(int(t) for t in outputs)


def _worker_main() -> int:
    """Child entry: generate once and print JSON token list to stdout."""
    import llaisys  # noqa: F401 — loads dual-path .so before generate
    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--device", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--layer_forward", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_tokens = _encode_prompt(tokenizer, args.prompt)
    tokens = _run_llaisys_greedy_inprocess(
        args.model,
        args.device,
        prompt_tokens,
        args.max_steps,
        layer_forward=bool(args.layer_forward),
    )
    print(json.dumps({"tokens": tokens, "env": os.environ.get(ENV_FLAG)}), flush=True)
    # Hard-exit after success: interpreter/CUDA teardown can SIGSEGV and
    # leave the device in a bad state for the next worker.
    os._exit(0)


def _run_llaisys_greedy_subprocess(
    model_path: str,
    device_name: str,
    prompt: str,
    max_new_tokens: int,
    layer_forward: bool,
) -> list[int]:
    env = os.environ.copy()
    env[ENV_FLAG] = "1" if layer_forward else "0"
    # Prefer worktree python package when parent already set PYTHONPATH.
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--worker",
        "--device",
        device_name,
        "--model",
        model_path,
        "--prompt",
        prompt,
        "--max_steps",
        str(max_new_tokens),
        "--layer_forward",
        "1" if layer_forward else "0",
    ]
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"worker layer_forward={layer_forward} failed rc={proc.returncode}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    # Last JSON line is the result (workers may print progress before it).
    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"worker produced no stdout\nstderr:\n{proc.stderr}")
    payload = json.loads(lines[-1])
    return list(int(t) for t in payload["tokens"])


def _run_hf_greedy(
    model_path: str,
    device_name: str,
    tokenizer,
    prompt_tokens: list[int],
    max_new_tokens: int,
) -> list[int]:
    import torch
    from transformers import AutoModelForCausalLM
    from test_utils import torch_device

    hf = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )
    try:
        inputs = torch.tensor([prompt_tokens], device=hf.device)
        with torch.no_grad():
            outputs = hf.generate(
                inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
        return outputs[0].tolist()
    finally:
        del hf
        gc.collect()
        if device_name == "nvidia" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


def _first_diff(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def main() -> int:
    # Worker mode must be detected before importing heavy parent-only deps.
    if "--worker" in sys.argv:
        return _worker_main()

    from transformers import AutoTokenizer
    from transformers import logging as hf_logging

    hf_logging.set_verbosity_error()

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max_steps", type=int, default=32)
    parser.add_argument(
        "--inprocess",
        action="store_true",
        help="Load both models in one process (may segfault on CUDA teardown)",
    )
    parser.add_argument(
        "--skip-hf",
        action="store_true",
        help="Skip optional HF diagnostic comparison",
    )
    args = parser.parse_args()

    if args.max_steps < 32:
        print(f"ERROR: --max_steps must be >= 32 (got {args.max_steps})", file=sys.stderr)
        return 2

    model_path = args.model
    if not os.path.isdir(model_path):
        print(f"ERROR: model path missing or incomplete: {model_path}", file=sys.stderr)
        return 2
    if not os.path.isfile(os.path.join(model_path, "model.safetensors")) and not any(
        Path(model_path).glob("*.safetensors")
    ):
        print(f"ERROR: no safetensors weights under {model_path}", file=sys.stderr)
        return 2

    # Do NOT import llaisys in the parent before subprocess workers: loading the
    # CUDA .so in-parent leaves driver state that can SIGSEGV the next worker.
    llaisys_pkg = Path(__file__).resolve().parents[1] / "python" / "llaisys" / "__init__.py"
    print(f"llaisys package (expected): {llaisys_pkg}")
    print(f"model: {model_path}")
    print(f"device: {args.device}")
    print(f"prompt: {args.prompt!r}")
    print(f"max_steps: {args.max_steps}")
    print(f"greedy: temperature=0.0 top_k=1 top_p=1.0")
    print(f"mode: {'inprocess' if args.inprocess else 'subprocess'}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    prompt_tokens = _encode_prompt(tokenizer, args.prompt)
    print(f"prompt_tokens ({len(prompt_tokens)}): {prompt_tokens}")

    t0 = time.time()
    print(f"\n=== LLAISYS legacy ({ENV_FLAG}=0) ===")
    if args.inprocess:
        legacy_tokens = _run_llaisys_greedy_inprocess(
            model_path, args.device, prompt_tokens, args.max_steps, layer_forward=False
        )
    else:
        legacy_tokens = _run_llaisys_greedy_subprocess(
            model_path, args.device, args.prompt, args.max_steps, layer_forward=False
        )
    print(f"tokens ({len(legacy_tokens)}): {legacy_tokens}")
    print(f"text: {tokenizer.decode(legacy_tokens, skip_special_tokens=True)}")
    print(f"elapsed: {time.time() - t0:.2f}s")

    t1 = time.time()
    print(f"\n=== LLAISYS layer ({ENV_FLAG}=1) ===")
    if args.inprocess:
        layer_tokens = _run_llaisys_greedy_inprocess(
            model_path, args.device, prompt_tokens, args.max_steps, layer_forward=True
        )
    else:
        # Brief pause so the prior worker's CUDA context can fully tear down.
        time.sleep(2.0)
        layer_tokens = _run_llaisys_greedy_subprocess(
            model_path, args.device, args.prompt, args.max_steps, layer_forward=True
        )
    print(f"tokens ({len(layer_tokens)}): {layer_tokens}")
    print(f"text: {tokenizer.decode(layer_tokens, skip_special_tokens=True)}")
    print(f"elapsed: {time.time() - t1:.2f}s")

    diff = _first_diff(legacy_tokens, layer_tokens)
    if diff is not None:
        print("\nHARD GATE FAILED: legacy vs layer token sequences differ")
        print(f"  first_diff_index={diff}")
        print(f"  legacy[{diff}]={legacy_tokens[diff] if diff < len(legacy_tokens) else '<eof>'}")
        print(f"  layer[{diff}]={layer_tokens[diff] if diff < len(layer_tokens) else '<eof>'}")
        return 1

    print("\nHARD GATE PASSED: legacy and layer greedy token sequences are identical")
    print(f"  equal_token_count={len(legacy_tokens)}")

    if not args.skip_hf:
        print("\n=== HF diagnostic (does not fail gate) ===")
        try:
            t2 = time.time()
            hf_tokens = _run_hf_greedy(
                model_path, args.device, tokenizer, prompt_tokens, args.max_steps
            )
            print(f"tokens ({len(hf_tokens)}): {hf_tokens}")
            print(f"text: {tokenizer.decode(hf_tokens, skip_special_tokens=True)}")
            print(f"elapsed: {time.time() - t2:.2f}s")
            hf_diff = _first_diff(legacy_tokens, hf_tokens)
            if hf_diff is None:
                print("HF diagnostic: identical to LLAISYS greedy tokens")
            else:
                print(f"HF diagnostic: differs from LLAISYS at index {hf_diff} (non-fatal)")
        except Exception as exc:  # noqa: BLE001 — diagnostic only
            print(f"HF diagnostic skipped due to error: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
