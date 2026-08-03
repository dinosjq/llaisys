"""Reusable inference shapes for model-representative operator benchmarks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InferenceShapeProfile:
    """Dimensions needed by linear and logits top-k benchmarks."""

    name: str
    hidden_size: int
    intermediate_size: int
    vocab_size: int

    @property
    def linear_variants(self) -> dict[str, tuple[int, int]]:
        """Return (out_features, in_features) for common Transformer projections."""
        return {
            "q_proj": (self.hidden_size, self.hidden_size),
            "gate_proj": (self.intermediate_size, self.hidden_size),
            "down_proj": (self.hidden_size, self.intermediate_size),
        }

    def topk_shapes(self, batch_sizes: tuple[int, ...] = (1, 16, 64, 128)) -> list[dict]:
        """Return logits-oriented top-k shapes plus representative hidden vectors."""
        return [
            {"shape": (self.hidden_size,), "k": 10},
            {"shape": (self.intermediate_size,), "k": 10},
            *({"shape": (batch_size, self.vocab_size), "k": 10}
              for batch_size in batch_sizes),
        ]


DEFAULT_SHAPE_PROFILES = {
    "deepseek-1.5b": InferenceShapeProfile(
        name="deepseek-1.5b", hidden_size=1536, intermediate_size=8960, vocab_size=151936),
    "llama-3.2-3b": InferenceShapeProfile(
        name="llama-3.2-3b", hidden_size=3072, intermediate_size=8192, vocab_size=128256),
}


def _config_path(model: str | Path) -> Path:
    path = Path(model)
    return path / "config.json" if path.is_dir() else path


def get_shape_profile(name: str, model: str | Path | None = None) -> InferenceShapeProfile:
    """Load a named default profile, optionally overriding dimensions from config.json."""
    default = DEFAULT_SHAPE_PROFILES[name]
    if model is None:
        return default

    config_path = _config_path(model)
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    required = ("hidden_size", "intermediate_size", "vocab_size")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"{config_path} is missing required fields: {', '.join(missing)}")
    return InferenceShapeProfile(
        name=default.name,
        hidden_size=int(config["hidden_size"]),
        intermediate_size=int(config["intermediate_size"]),
        vocab_size=int(config["vocab_size"]),
    )
