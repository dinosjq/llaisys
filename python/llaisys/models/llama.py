"""Llama-3.x vertical slice on the shared decoder runtime + Llama layer stack.

Sets ``LLAISYS_LAYER_STACK=llama`` before creating the C++ model so forward uses
``run_llama_layer_stack`` (no-bias Attn/FFN). Scheduler / Sequence / BlockManager
are unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import safetensors
from ctypes import c_float, c_int, c_int64, c_size_t, c_void_p, byref

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysQwen2Meta,
    LlaisysQwen2Weights,
    llaisysQwen2Model_t,
)
from ..tensor import Tensor
from .llama_weight_map import assign_to_legacy_weights, resolve_hf_weight


class Llama:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        # Select Llama layer stack before C++ construction reads the env.
        os.environ["LLAISYS_LAYER_STACK"] = "llama"
        os.environ.pop("LLAISYS_QWEN2_LAYER_FORWARD", None)

        model_path = Path(model_path)
        self._device = device
        self._tensors = []
        self._closed = False

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        eos = config.get("eos_token_id", config.get("eos_token_ids", None))
        if isinstance(eos, list):
            # Prefer the chat end-of-turn id when present (Llama-3 Instruct: 128009).
            eos_token = int(eos[-1]) if eos else 0
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

        meta = LlaisysQwen2Meta()
        meta.dtype = int(dtype)
        meta.nlayer = int(config["num_hidden_layers"])
        meta.hs = int(config["hidden_size"])
        meta.nh = int(config["num_attention_heads"])
        meta.nkvh = int(config.get("num_key_value_heads", meta.nh))
        head_dim = config.get("head_dim")
        meta.dh = int(head_dim) if head_dim is not None else int(meta.hs // meta.nh)
        meta.di = int(config["intermediate_size"])
        # Cap positional metadata; KV capacity still comes from config.hpp.
        meta.maxseq = min(int(config.get("max_position_embeddings", 8192)), 8192)
        meta.voc = int(config["vocab_size"])
        meta.epsilon = float(config.get("rms_norm_eps", 1e-5))
        meta.theta = float(config.get("rope_theta", 500000.0))
        meta.end_token = int(eos_token)

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

        self._meta = meta
        device_ids = (c_int * 1)(0)
        self._model: llaisysQwen2Model_t = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta),
            int(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create Llama model (shared Qwen2 runtime)")

        weights_ptr = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model)
        if not weights_ptr:
            raise RuntimeError("Failed to get model weights")
        self._weights: LlaisysQwen2Weights = weights_ptr.contents

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    arr = data_.get_tensor(name_)
                    self._assign_weight(name_, arr)

        # tie_word_embeddings
        if not self._weights.out_embed:
            self._weights.out_embed = self._weights.in_embed

    def close(self):
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysQwen2ModelDestroy(self._model)
            self._model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

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

    def _assign_weight(self, name: str, arr: np.ndarray):
        resolved = resolve_hf_weight(name)
        if resolved is None:
            return
        role, layer = resolved
        t = self._tensor_from_numpy(arr)
        assign_to_legacy_weights(self._weights, role, layer, t.lib_tensor())

    def _submit_request(self, tokens, max_new_tokens, top_k, top_p, temperature):
        arr = (c_int64 * len(tokens))(*tokens)
        request = LIB_LLAISYS.llaisysQwen2RequestSubmit(
            self._model,
            arr,
            c_size_t(len(tokens)),
            c_int64(max_new_tokens),
            c_int(top_k),
            c_float(top_p),
            c_float(temperature),
        )
        if not request:
            raise RuntimeError("llaisysQwen2RequestSubmit failed")
        return request

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
        callback: Callable[[int, int, tuple[int, ...]], bool] = None,
    ) -> list[int]:
        if max_new_tokens is None:
            max_new_tokens = -1

        _loop_bound = max_new_tokens if max_new_tokens >= 0 else 128
        tokens = list(int(t) for t in inputs)
        request = self._submit_request(tokens, max_new_tokens, top_k, top_p, temperature)
        try:
            for step in range(_loop_bound):
                next_token = int(LIB_LLAISYS.llaisysQwen2RequestAwait(request))
                if next_token == -2:
                    break
                if next_token == -1:
                    raise RuntimeError("llaisysQwen2RequestAwait failed")
                tokens.append(next_token)
                if callback is not None and callback(next_token, step, tuple(tokens)) is False:
                    LIB_LLAISYS.llaisysQwen2RequestAbort(request)
                    break
                if next_token == int(self._meta.end_token):
                    break
        finally:
            LIB_LLAISYS.llaisysQwen2RequestRelease(request)
        return tokens
