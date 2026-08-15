"""Llama-3.x on shared decoder runtime + Llama layer assembly in C++ Llama::forward."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import safetensors
from ctypes import c_float, c_int, c_int64, c_size_t, c_void_p, byref

from ..libllaisys import (
    LIB_LLAISYS,
    DeviceType,
    DataType,
    LlaisysModelArch,
    LlaisysModelMeta,
    llaisysModel_t,
)
from ..tensor import Tensor
from .llama_weight_map import resolve_hf_weight
from .qwen2_weight_map import WeightRole


class Llama:
    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)
        self._device = device
        self._tensors = []
        self._closed = False
        self._in_embed = None
        self._out_embed_set = False

        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"config.json not found in {model_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        eos = config.get("eos_token_id", config.get("eos_token_ids", None))
        if isinstance(eos, list):
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

        meta = LlaisysModelMeta()
        meta.dtype = int(dtype)
        meta.nlayer = int(config["num_hidden_layers"])
        meta.hs = int(config["hidden_size"])
        meta.nh = int(config["num_attention_heads"])
        meta.nkvh = int(config.get("num_key_value_heads", meta.nh))
        head_dim = config.get("head_dim")
        meta.dh = int(head_dim) if head_dim is not None else int(meta.hs // meta.nh)
        meta.di = int(config["intermediate_size"])
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
        self._model: llaisysModel_t = LIB_LLAISYS.llaisysModelCreate(
            int(LlaisysModelArch.LLAMA),
            byref(meta),
            int(device),
            device_ids,
            1,
        )
        if not self._model:
            raise RuntimeError("Failed to create Llama model")

        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.safe_open(file, framework="numpy", device="cpu") as data_:
                for name_ in data_.keys():
                    arr = data_.get_tensor(name_)
                    self._assign_weight(name_, arr)

        if not self._out_embed_set and self._in_embed is not None:
            LIB_LLAISYS.llaisysModelSetWeight(
                self._model,
                int(WeightRole.OutEmbed),
                c_size_t(0),
                self._in_embed.lib_tensor(),
            )

    def close(self):
        if self._closed:
            return
        self._closed = True
        if getattr(self, "_model", None):
            LIB_LLAISYS.llaisysModelDestroy(self._model)
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
        if role == WeightRole.InEmbed:
            self._in_embed = t
        if role == WeightRole.OutEmbed:
            self._out_embed_set = True
        LIB_LLAISYS.llaisysModelSetWeight(
            self._model,
            int(role),
            c_size_t(layer),
            t.lib_tensor(),
        )

    def _submit_request(self, tokens, max_new_tokens, top_k, top_p, temperature):
        arr = (c_int64 * len(tokens))(*tokens)
        request = LIB_LLAISYS.llaisysModelRequestSubmit(
            self._model,
            arr,
            c_size_t(len(tokens)),
            c_int64(max_new_tokens),
            c_int(top_k),
            c_float(top_p),
            c_float(temperature),
        )
        if not request:
            raise RuntimeError("llaisysModelRequestSubmit failed")
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
                next_token = int(LIB_LLAISYS.llaisysModelRequestAwait(request))
                if next_token == -2:
                    break
                if next_token == -1:
                    raise RuntimeError("llaisysModelRequestAwait failed")
                tokens.append(next_token)
                if callback is not None and callback(next_token, step, tuple(tokens)) is False:
                    LIB_LLAISYS.llaisysModelRequestAbort(request)
                    break
                if next_token == int(self._meta.end_token):
                    break
        finally:
            LIB_LLAISYS.llaisysModelRequestRelease(request)
        return tokens
