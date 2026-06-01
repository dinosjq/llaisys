import gc
from test_utils import *
import warnings

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import logging as hf_logging
import torch
from huggingface_hub import snapshot_download
import os
import time
import llaisys
import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
hf_logging.set_verbosity_error()
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated! Use `dtype` instead!")

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
        device_map=torch_device(device_name),
        trust_remote_code=True,
    )

    return tokenizer, model, model_path


def hf_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
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


def llaisys_infer(
    prompt, tokenizer, model, max_new_tokens=128, top_p=0.8, top_k=50, temperature=0.8
):
    input_content = tokenizer.apply_chat_template(
        conversation=[{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer.encode(input_content)
    outputs = model.generate(
        inputs,
        max_new_tokens=max_new_tokens,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
    )

    return outputs, tokenizer.decode(outputs, skip_special_tokens=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--model", default=None, type=str)
    parser.add_argument("--prompt", default="""给定一个仅由大小写英文字母和下划线组成的字符串 
S, 表示对着魔镜呼唤的名字. 请你编写一个程序, 模拟魔镜的反馈机制，规则如下: 
如果 S 恰好为 awdec, 请输出 Fantasy_Blue;
如果 S 恰好为 Fantasy_Blue, 请输出 awdec;
对于其他任何名字(即 S 为上述两者之外的字符串),魔镜都无法产生特殊共鸣，请输出 other.
输入描述:
输入仅包含一行一个字符串 S (1 ≤ |S| ≤ 100), 表示呼唤的名字.
保证 S 仅由大写英文字母、小写英文字母和下划线组成.
输出描述:
输出仅包含一行一个字符串, 表示魔镜反馈的结果(Fantasy_Blue/awdec 或 other. 
做一下这道题 使用 cpp""", type=str)
    """Who are you?"""
    parser.add_argument("--max_steps", default=2048, type=int)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--top_k", default=50, type=int)
    parser.add_argument("--temperature", default=1.0, type=float)
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()

    top_p, top_k, temperature = args.top_p, args.top_k, args.temperature
    if args.test:
        top_p, top_k, temperature = 1.0, 1, 1.0
    tokenizer, model, model_path = load_hf_model(args.model, args.device)

    # Example prompt
    start_time = time.time()
    tokens, output = hf_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )
    end_time = time.time()

    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    print("\n=== Answer ===\n")
    # print("Tokens:")
    # print(tokens)
    print("\nContents:")
    print(output)
    print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")

    model = load_llaisys_model(model_path, args.device)

    start_time = time.time()
    
    llaisys_tokens, llaisys_output = llaisys_infer(
        args.prompt,
        tokenizer,
        model,
        max_new_tokens=args.max_steps,
        top_p=top_p,
        top_k=top_k,
        temperature=temperature,
    )

    end_time = time.time()

    print("\n=== Your Result ===\n")
    # print("Tokens:")
    # print(llaisys_tokens)
    print("\nContents:")
    print(llaisys_output)
    # print("\n")
    print(f"Time elapsed: {(end_time - start_time):.2f}s\n")
    print("\n")

    # if args.test:
    #     assert llaisys_tokens == tokens
    #     print("\033[92mTest passed!\033[0m\n")

    # 显式释放 C++ 模型，避免依赖解释器退出阶段的 __del__ 导致后台线程析构竞态。
    if hasattr(model, "close"):
        model.close()
    del model
    gc.collect()
