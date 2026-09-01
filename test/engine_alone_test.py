"""引擎单独跑：只加载 LLAISYS + HF tokenizer(CPU)，不做 HF GPU 推理。
验证 flash_decoding 集成后引擎能否独立跑通（隔离 test_infer 的 HF-stream 交互）。"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, llaisys
from transformers import AutoTokenizer
from test_utils import llaisys_device

def main():
    model_path = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.load_causal_lm(model_path, llaisys_device("nvidia"))
    prompt = "用一句话介绍你自己。"
    inputs = tokenizer.encode(prompt)
    outputs = model.generate(inputs, max_new_tokens=32, top_k=50, top_p=0.8, temperature=0.8)
    print("=== LLS 引擎单独输出 ===")
    print(tokenizer.decode(outputs, skip_special_tokens=True))
    if hasattr(model, "close"):
        model.close()
    print("ENGINE_ALONE_OK")

if __name__ == "__main__":
    main()
