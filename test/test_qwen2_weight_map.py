"""Unit tests for HF name → (WeightRole, layer) mapping (Migration Task 4).

Covers DeepSeek-R1-Distill-Qwen-1.5B critical keys including Q/K/V bias and
lm_head / embed tie fallback documentation.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PYTHON = _REPO / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from llaisys.models.qwen2_weight_map import (  # noqa: E402
    WeightRole,
    resolve_hf_weight,
    legacy_slot_for_role,
)


# Representative DeepSeek-Qwen2 keys (layer 0 + globals). Full checkpoint has
# the same suffixes on every layer; bias present on q/k/v only.
DEEPSEEK_CRITICAL_KEYS = {
    "model.embed_tokens.weight": (WeightRole.InEmbed, 0),
    "lm_head.weight": (WeightRole.OutEmbed, 0),
    "model.norm.weight": (WeightRole.OutNorm, 0),
    "model.layers.0.input_layernorm.weight": (WeightRole.AttnNorm, 0),
    "model.layers.0.self_attn.q_proj.weight": (WeightRole.AttnQ_W, 0),
    "model.layers.0.self_attn.q_proj.bias": (WeightRole.AttnQ_B, 0),
    "model.layers.0.self_attn.k_proj.weight": (WeightRole.AttnK_W, 0),
    "model.layers.0.self_attn.k_proj.bias": (WeightRole.AttnK_B, 0),
    "model.layers.0.self_attn.v_proj.weight": (WeightRole.AttnV_W, 0),
    "model.layers.0.self_attn.v_proj.bias": (WeightRole.AttnV_B, 0),
    "model.layers.0.self_attn.o_proj.weight": (WeightRole.AttnO_W, 0),
    "model.layers.0.post_attention_layernorm.weight": (WeightRole.MlpNorm, 0),
    "model.layers.0.mlp.gate_proj.weight": (WeightRole.MlpGate_W, 0),
    "model.layers.0.mlp.up_proj.weight": (WeightRole.MlpUp_W, 0),
    "model.layers.0.mlp.down_proj.weight": (WeightRole.MlpDown_W, 0),
    # Non-zero layer index parsing
    "model.layers.27.self_attn.q_proj.bias": (WeightRole.AttnQ_B, 27),
}


class TestQwen2WeightMap(unittest.TestCase):
    def test_deepseek_critical_keys_resolve(self):
        for name, expected in DEEPSEEK_CRITICAL_KEYS.items():
            with self.subTest(name=name):
                self.assertEqual(resolve_hf_weight(name), expected)

    def test_unknown_key_returns_none(self):
        self.assertIsNone(resolve_hf_weight("model.layers.0.self_attn.q_norm.weight"))
        self.assertIsNone(resolve_hf_weight("not.a.weight"))

    def test_legacy_slot_covers_all_roles(self):
        """Every WeightRole maps to a LlaisysQwen2Weights field name."""
        for role in WeightRole:
            slot = legacy_slot_for_role(role)
            self.assertIsInstance(slot, str)
            self.assertTrue(slot)

    def test_tie_fallback_documented(self):
        """Models without lm_head.weight keep OutEmbed unset; loader ties to InEmbed."""
        self.assertEqual(
            resolve_hf_weight("model.embed_tokens.weight"),
            (WeightRole.InEmbed, 0),
        )
        # Absence of lm_head is intentional: assign path leaves out_embed None
        # until qwen2.Qwen2 ties out_embed = in_embed after safetensors load.


if __name__ == "__main__":
    unittest.main()
