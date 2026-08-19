"""Load a LLAISYS causal LM from a HuggingFace model directory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Union

from ..libllaisys import DeviceType
from .llama import Llama
from .qwen2 import Qwen2

CausalLM = Union[Qwen2, Llama]


def read_model_config(model_path: str | Path) -> dict[str, Any]:
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def detect_model_arch(model_path: str | Path) -> str:
    """Return ``\"qwen2\"`` or ``\"llama\"`` from HF config."""
    config = read_model_config(model_path)
    archs = [str(a).lower() for a in (config.get("architectures") or [])]
    model_type = str(config.get("model_type", "")).lower()
    joined = " ".join(archs)

    if "llama" in joined or model_type == "llama":
        return "llama"
    if "qwen2" in joined or model_type in ("qwen2", "qwen"):
        return "qwen2"
    if any("qwen" in a for a in archs):
        return "qwen2"
    raise ValueError(
        f"Unsupported model architecture: architectures={archs!r} model_type={model_type!r}"
    )


def load_causal_lm(
    model_path: str | Path,
    device: DeviceType = DeviceType.CPU,
    *,
    quantize_weights: bool = False,
) -> CausalLM:
    arch = detect_model_arch(model_path)
    if arch == "llama":
        return Llama(model_path, device, quantize_weights=quantize_weights)
    return Qwen2(model_path, device, quantize_weights=quantize_weights)


__all__ = [
    "CausalLM",
    "detect_model_arch",
    "load_causal_lm",
    "read_model_config",
]
