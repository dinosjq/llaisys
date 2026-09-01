"""引擎端到端：测多个短 prompt（<16 token，触发 TC prefill 越界槽位路径）。"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch, llaisys
from transformers import AutoTokenizer
from test_utils import llaisys_device

def run(prompt):
    model_path = os.path.expanduser("~/models/DeepSeek-R1-Distill-Qwen-1.5B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = llaisys.models.load_causal_lm(model_path, llaisys_device("nvidia"))
    inputs = tokenizer.encode(prompt)
    print(f"--- prompt={prompt!r} (in_tokens={len(inputs)})")
    outputs = model.generate(inputs, max_new_tokens=24, top_k=50, top_p=0.8, temperature=0.8)
    print("   out:", tokenizer.decode(outputs, skip_special_tokens=True))
    if hasattr(model, "close"):
        model.close()
    del model
    torch.cuda.empty_cache()

if __name__ == "__main__":
    for p in ["hi", "hello", "What is 2+2?"]:
        run(p)
    print("ENGINE_SHORT_DONE")
