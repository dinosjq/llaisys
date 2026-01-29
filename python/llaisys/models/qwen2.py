from typing import Sequence
from pathlib import Path
import json
import numpy as np
import safetensors
from ctypes import c_int64, c_size_t, c_int, c_void_p, POINTER, byref

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    llaisysDeviceType_t,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
)
from ..tensor import Tensor


class Qwen2:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # 初始化设备 路径
        model_path = Path(model_path)
        self._device = device
        self._tensors = []

        # model 的 config 路径
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        # 结束符 token_id
        eos = config.get("eos_token_id", config.get("eos_token_ids", None))
        if isinstance(eos, list):
            eos_token = int(eos[0]) if eos else 0
        elif eos is None:
            eos_token = 0
        else:
            eos_token = int(eos)

        dtype_name = str(config.get("torch_dtype", "bfloat16")).lower()
        if "bfloat" in dtype_name:
            dtype = DataType.BF16
        elif "float16" in dtype_name or "fp16" in dtype_name:
            dtype = DataType.F16
        else:
            dtype = DataType.F32

        # 加载模型其他参数配置
        meta = LlaisysQwen2Meta()
        meta.dtype = int(dtype)
        meta.nlayer = int(config["num_hidden_layers"])
        meta.hs = int(config["hidden_size"])
        meta.nh = int(config["num_attention_heads"])
        meta.nkvh = int(config.get("num_key_value_heads", meta.nh))
        meta.dh = int(meta.hs // meta.nh)
        meta.di = int(config["intermediate_size"])
        meta.maxseq = int(config.get("max_position_embeddings", config.get("max_seq_len", 0)))
        meta.voc = int(config["vocab_size"])
        meta.epsilon = float(config.get("rms_norm_eps", 1e-6))
        meta.theta = float(config.get("rope_theta", 10000.0))
        meta.end_token = int(eos_token)

        # Register bfloat16 dtype for numpy via ml_dtypes
        try:
            import ml_dtypes  # noqa: F401
            if "bfloat16" not in np.sctypeDict:
                np.sctypeDict["bfloat16"] = ml_dtypes.bfloat16
            if "bf16" not in np.sctypeDict:
                np.sctypeDict["bf16"] = ml_dtypes.bfloat16
        except Exception as exc:
            raise RuntimeError(
                "Loading bfloat16 weights requires ml_dtypes. Please install ml_dtypes."
            ) from exc

        # 设置 meta
        self._meta = meta
        # 通过 create 创建 model
        self._model: llaisysQwen2Model_t = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            int(device),
            None,
            0,
        )
        if not self._model:
            raise RuntimeError("Failed to create Qwen2 model")

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_ptr:
            raise RuntimeError("Failed to get Qwen2 weights")
        self._weights: LlaisysQwen2Weights = weights_ptr.contents

        # 从 safetensors 中加载模型参数
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    arr = data_.get_tensor(name_)
                    self._assign_weight(name_, arr)

        # tie weights if needed
        if not self._weights.out_embed:
            self._weights.out_embed = self._weights.in_embed

    # 加载 tensor: 将 numpy 转为 tensor
    def _tensor_from_numpy(self, arr: np.ndarray) -> Tensor:
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32:
            dtype = DataType.F32
        elif arr.dtype == np.float16:
            dtype = DataType.F16
        elif str(arr.dtype) == "bfloat16":
            dtype = DataType.BF16
        elif arr.dtype == np.int64:
            dtype = DataType.I64
        else:
            raise ValueError(f"Unsupported dtype: {arr.dtype}")

        t = Tensor(arr.shape, dtype=dtype, device=self._device)
        t.load(c_void_p(arr.ctypes.data))
        self._tensors.append(t)
        return t

    # 加载权重: 将对应的 tensor 赋给对应的权重
    def _assign_weight(self, name: str, arr: np.ndarray):
        t = self._tensor_from_numpy(arr)
        if name == "model.embed_tokens.weight":
            self._weights.in_embed = t.lib_tensor()
            return
        if name == "lm_head.weight":
            self._weights.out_embed = t.lib_tensor()
            return
        if name == "model.norm.weight":
            self._weights.out_norm_w = t.lib_tensor()
            return

        if name.startswith("model.layers."):
            parts = name.split(".")
            if len(parts) < 4:
                return
            layer = int(parts[2])
            suffix = ".".join(parts[3:])
            if suffix == "input_layernorm.weight":
                self._weights.attn_norm_w[layer] = t.lib_tensor()
            elif suffix == "self_attn.q_proj.weight":
                self._weights.attn_q_w[layer] = t.lib_tensor()
            elif suffix == "self_attn.q_proj.bias":
                self._weights.attn_q_b[layer] = t.lib_tensor()
            elif suffix == "self_attn.k_proj.weight":
                self._weights.attn_k_w[layer] = t.lib_tensor()
            elif suffix == "self_attn.k_proj.bias":
                self._weights.attn_k_b[layer] = t.lib_tensor()
            elif suffix == "self_attn.v_proj.weight":
                self._weights.attn_v_w[layer] = t.lib_tensor()
            elif suffix == "self_attn.v_proj.bias":
                self._weights.attn_v_b[layer] = t.lib_tensor()
            elif suffix == "self_attn.o_proj.weight":
                self._weights.attn_o_w[layer] = t.lib_tensor()
            elif suffix == "post_attention_layernorm.weight":
                self._weights.mlp_norm_w[layer] = t.lib_tensor()
            elif suffix == "mlp.gate_proj.weight":
                self._weights.mlp_gate_w[layer] = t.lib_tensor()
            elif suffix == "mlp.up_proj.weight":
                self._weights.mlp_up_w[layer] = t.lib_tensor()
            elif suffix == "mlp.down_proj.weight":
                self._weights.mlp_down_w[layer] = t.lib_tensor()

    # 生成回复
    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        if max_new_tokens is None:
            max_new_tokens = 128

        tokens = list(int(t) for t in inputs)
        for _ in range(max_new_tokens):
            arr = (c_int64 * len(tokens))(*tokens)
            next_token = int(
                LIB_LLAISYS.llaisysQwen2ModelInfer(
                    self._model,
                    arr,
                    c_size_t(len(tokens)),
                )
            )
            tokens.append(next_token)
            if next_token == int(self._meta.end_token):
                break
        return tokens
