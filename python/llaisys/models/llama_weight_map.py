"""HF safetensors name → (WeightRole, layer) for Llama-3.x (no attention/MLP bias).

Upload still fills ``LlaisysQwen2Weights`` (shared decoder runtime); C++
``sync_weights_from_legacy_struct`` + ``LLAISYS_LAYER_STACK=llama`` selects
``run_llama_layer_stack``.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .qwen2_weight_map import (
    WeightRole,
    assign_to_legacy_weights,
    is_layered_role,
    legacy_slot_for_role,
)

_GLOBAL_HF_TO_ROLE: dict[str, WeightRole] = {
    "model.embed_tokens.weight": WeightRole.InEmbed,
    "lm_head.weight": WeightRole.OutEmbed,  # optional; often tied
    "model.norm.weight": WeightRole.OutNorm,
}

_LAYER_SUFFIX_TO_ROLE: dict[str, WeightRole] = {
    "input_layernorm.weight": WeightRole.AttnNorm,
    "self_attn.q_proj.weight": WeightRole.AttnQ_W,
    "self_attn.k_proj.weight": WeightRole.AttnK_W,
    "self_attn.v_proj.weight": WeightRole.AttnV_W,
    "self_attn.o_proj.weight": WeightRole.AttnO_W,
    "post_attention_layernorm.weight": WeightRole.MlpNorm,
    "mlp.gate_proj.weight": WeightRole.MlpGate_W,
    "mlp.up_proj.weight": WeightRole.MlpUp_W,
    "mlp.down_proj.weight": WeightRole.MlpDown_W,
}


def resolve_hf_weight(name: str) -> Optional[Tuple[WeightRole, int]]:
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


__all__ = [
    "WeightRole",
    "resolve_hf_weight",
    "assign_to_legacy_weights",
    "is_layered_role",
    "legacy_slot_for_role",
]
