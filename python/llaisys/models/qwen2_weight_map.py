"""HF safetensors name → (WeightRole, layer) for Qwen2 upload via SetWeight."""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple


class WeightRole(IntEnum):
    """Mirrors ``llaisys::core::WeightRole`` / ``LlaisysWeightRole``."""

    InEmbed = 0
    OutEmbed = 1
    OutNorm = 2
    AttnNorm = 3
    AttnQ_W = 4
    AttnQ_B = 5
    AttnK_W = 6
    AttnK_B = 7
    AttnV_W = 8
    AttnV_B = 9
    AttnO_W = 10
    MlpNorm = 11
    MlpGate_W = 12
    MlpUp_W = 13
    MlpDown_W = 14


_GLOBAL_HF_TO_ROLE: dict[str, WeightRole] = {
    "model.embed_tokens.weight": WeightRole.InEmbed,
    "lm_head.weight": WeightRole.OutEmbed,
    "model.norm.weight": WeightRole.OutNorm,
}

_LAYER_SUFFIX_TO_ROLE: dict[str, WeightRole] = {
    "input_layernorm.weight": WeightRole.AttnNorm,
    "self_attn.q_proj.weight": WeightRole.AttnQ_W,
    "self_attn.q_proj.bias": WeightRole.AttnQ_B,
    "self_attn.k_proj.weight": WeightRole.AttnK_W,
    "self_attn.k_proj.bias": WeightRole.AttnK_B,
    "self_attn.v_proj.weight": WeightRole.AttnV_W,
    "self_attn.v_proj.bias": WeightRole.AttnV_B,
    "self_attn.o_proj.weight": WeightRole.AttnO_W,
    "post_attention_layernorm.weight": WeightRole.MlpNorm,
    "mlp.gate_proj.weight": WeightRole.MlpGate_W,
    "mlp.up_proj.weight": WeightRole.MlpUp_W,
    "mlp.down_proj.weight": WeightRole.MlpDown_W,
}


def resolve_hf_weight(name: str) -> Optional[Tuple[WeightRole, int]]:
    """Map a HuggingFace / safetensors tensor name to ``(WeightRole, layer)``.

    Returns ``None`` for unrecognized names (silently skipped by the loader).
    Global roles use ``layer=0``.
    """
    role = _GLOBAL_HF_TO_ROLE.get(name)
    if role is not None:
        return role, 0

    if not name.startswith("model.layers."):
        return None
    parts = name.split(".")
    if len(parts) < 4:
        return None
    try:
        layer = int(parts[2])
    except ValueError:
        return None
    suffix = ".".join(parts[3:])
    role = _LAYER_SUFFIX_TO_ROLE.get(suffix)
    if role is None:
        return None
    return role, layer
