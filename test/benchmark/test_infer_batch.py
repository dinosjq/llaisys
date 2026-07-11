import threading
import argparse
import time
import warnings

import gc
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
from huggingface_hub import snapshot_download
import torch
import llaisys
import os

hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")

# inline minimal helpers from test/test_infer.py to avoid import issues
def load_hf_model(model_path=None, device_name="cpu"):
    model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    if model_path and os.path.isdir(model_path):
        print(f"Loading model from local path: {model_path}")
    else:
        print(f"Loading model from Hugging Face: {model_id}")
        model_path = snapshot_download(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": torch_device(device_name)} if device_name == "cpu" else None,
        trust_remote_code=True,
    )
    return tokenizer, model, model_path


def hf_infer(prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer.encode(input_content, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=max_new_tokens,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            use_cache=True,
        )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return outputs[0].tolist(), result


def load_llaisys_model(model_path, device_name):
    model = llaisys.models.Qwen2(model_path, llaisys_device(device_name))
    return model


def llaisys_infer(prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=False
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs, max_new_tokens=max_new_tokens, top_k=top_k, top_p=top_p, temperature=temperature
    )
    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)

def torch_device(device_name: str):
    if device_name == "cpu":
        return torch.device("cpu")
    elif device_name == "nvidia":
        return torch.device("cuda:0")
    else:
        raise ValueError(f"Unsupported device name: {device_name}")

def llaisys_device(device_name: str):
    if device_name == "cpu":
        return llaisys.DeviceType.CPU
    elif device_name == "nvidia":
        return llaisys.DeviceType.NVIDIA
    else:
        raise ValueError(f"Unsupported device name: {device_name}")


def run_single_hf(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature, results, idx):
    tokens, text = hf_infer(prompt, tokenizer, model, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, temperature=temperature)
    results[idx] = (tokens, text)


def run_single_ll(prompt, tokenizer, model, max_new_tokens, top_p, top_k, temperature, results, idx):
    tokens, text = llaisys_infer(prompt, tokenizer, model, max_new_tokens=max_new_tokens, top_p=top_p, top_k=top_k, temperature=temperature)
    results[idx] = (tokens, text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="nvidia", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--max_steps", default=32, type=int)
    parser.add_argument("--top_p", default=1.0, type=float)
    parser.add_argument("--top_k", default=1, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        args.top_p = 1.0
        args.top_k = 1
        args.temperature = 1.0

    tokenizer, hf_model, model_path = load_hf_model(args.model, args.device)
    # prepare multiple prompts to exercise batching
    prompts = [
        "Hello, how are you?",
        "What is the capital of France?",
        "Summarize the following: Machine learning is...",
        "Write a short poem about AI."
    ]

    max_new_tokens = args.max_steps
    top_p = args.top_p
    top_k = args.top_k
    temperature = args.temperature

    # run hf sequentially to get references
    print("\n=== Answer ===\n")
    print("Running HF reference inference for prompts...")
    hf_results = [None] * len(prompts)
    hf_start = time.time()
    for i, p in enumerate(prompts):
        run_single_hf(p, tokenizer, hf_model, max_new_tokens, top_p, top_k, temperature, hf_results, i)
        print(f"  HF [{i}] tokens={len(hf_results[i][0])} text_len={len(hf_results[i][1])}")
    hf_end = time.time()

    print(f"HF reference finished in {hf_end - hf_start:.3f}s")

    # load llaisys model
    print("Loading LLAISYS model...")
    ll_model = load_llaisys_model(model_path, args.device)

    # run llaisys concurrently to exercise batching
    print("\n=== Your Result ===\n")
    print("Running LLAISYS batched (concurrent) inference for prompts...")
    ll_results = [None] * len(prompts)
    threads = []
    start = time.time()
    for i, p in enumerate(prompts):
        t = threading.Thread(target=run_single_ll, args=(p, tokenizer, ll_model, max_new_tokens, top_p, top_k, temperature, ll_results, i))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
    end = time.time()

    print(f"LLAISYS batch inference finished in {end-start:.3f}s")
    for i, p in enumerate(prompts):
        hf_tokens, hf_text = hf_results[i]
        ll_tokens, ll_text = ll_results[i]
        ok = hf_tokens == ll_tokens
        print(f"  Prompt [{i}] ok={ok} hf_len={len(hf_tokens)} ll_len={len(ll_tokens)}")
        if not ok:
            print("    HF text:\n", hf_text)
            print("    LL text:\n", ll_text)

    if all(hf_tokens == ll_tokens for (hf_tokens, _), (ll_tokens, _) in zip(hf_results, ll_results)):
        print("\n\033[92mBatch test passed!\033[0m\n")
    else:
        raise SystemExit("Batch test failed: outputs differ")
