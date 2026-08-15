"""HF safetensors name → (WeightRole, layer) for Qwen2 (compat shell / Scheme A).

Python still fills ``LlaisysQwen2Weights``; C++ copies into ``ModelContext`` via
``sync_weights_from_legacy_struct()`` / ``Model::set_weight``. This module is the
documented upload-boundary map and drives ``Qwen2._assign_weight``.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Optional, Tuple


class WeightRole(IntEnum):
    """Mirrors ``llaisys::framework::WeightRole`` (upload face D)."""

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


# Global (non-layer) HF names → role. layer index is ignored by set_weight.
_GLOBAL_HF_TO_ROLE: dict[str, WeightRole] = {
    "model.embed_tokens.weight": WeightRole.InEmbed,
    "lm_head.weight": WeightRole.OutEmbed,
    "model.norm.weight": WeightRole.OutNorm,
}

# ``model.layers.{i}.`` + suffix → per-layer role.
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

# WeightRole → attribute on ``LlaisysQwen2Weights`` (pointer arrays use [layer]).
_ROLE_TO_LEGACY_SLOT: dict[WeightRole, str] = {
    WeightRole.InEmbed: "in_embed",
    WeightRole.OutEmbed: "out_embed",
    WeightRole.OutNorm: "out_norm_w",
    WeightRole.AttnNorm: "attn_norm_w",
    WeightRole.AttnQ_W: "attn_q_w",
    WeightRole.AttnQ_B: "attn_q_b",
    WeightRole.AttnK_W: "attn_k_w",
    WeightRole.AttnK_B: "attn_k_b",
    WeightRole.AttnV_W: "attn_v_w",
    WeightRole.AttnV_B: "attn_v_b",
    WeightRole.AttnO_W: "attn_o_w",
    WeightRole.MlpNorm: "mlp_norm_w",
    WeightRole.MlpGate_W: "mlp_gate_w",
    WeightRole.MlpUp_W: "mlp_up_w",
    WeightRole.MlpDown_W: "mlp_down_w",
}

_LAYERED_ROLES = frozenset(
    {
        WeightRole.AttnNorm,
        WeightRole.AttnQ_W,
        WeightRole.AttnQ_B,
        WeightRole.AttnK_W,
        WeightRole.AttnK_B,
        WeightRole.AttnV_W,
        WeightRole.AttnV_B,
        WeightRole.AttnO_W,
        WeightRole.MlpNorm,
        WeightRole.MlpGate_W,
        WeightRole.MlpUp_W,
        WeightRole.MlpDown_W,
    }
)


def resolve_hf_weight(name: str) -> Optional[Tuple[WeightRole, int]]:
    """Map a HuggingFace / safetensors tensor name to ``(WeightRole, layer)``.

    Returns ``None`` for unrecognized names (silently skipped by the loader).
    Global roles use ``layer=0`` (ignored by C++ ``set_weight``).
    """
    role = _GLOBAL_HF_TO_ROLE.get(name)
    if role is not None:
        return role, 0

    if not name.startswith("model.layers."):
        return None
    parts = name.split(".")
    # model.layers.{idx}.{suffix...}
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


def legacy_slot_for_role(role: WeightRole) -> str:
    """LlaisysQwen2Weights field name for a role (documentation + assign path)."""
    return _ROLE_TO_LEGACY_SLOT[role]


def is_layered_role(role: WeightRole) -> bool:
    return role in _LAYERED_ROLES


def assign_to_legacy_weights(weights, role: WeightRole, layer: int, lib_tensor) -> None:
    """Write ``lib_tensor`` into the matching ``LlaisysQwen2Weights`` slot."""
    slot = legacy_slot_for_role(role)
    if is_layered_role(role):
        getattr(weights, slot)[layer] = lib_tensor
    else:
        setattr(weights, slot, lib_tensor)
