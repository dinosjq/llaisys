"""Unit tests for HF name → (WeightRole, layer) mapping."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_PYTHON = _REPO / "python"
if str(_PYTHON) not in sys.path:
    sys.path.insert(0, str(_PYTHON))

from llaisys.models import qwen2_weight_map as qwen2_map  # noqa: E402
from llaisys.models import llama_weight_map as llama_map  # noqa: E402
from llaisys.models.qwen2_weight_map import WeightRole  # noqa: E402


QWEN2_CRITICAL_KEYS = {
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
    "model.layers.27.self_attn.q_proj.bias": (WeightRole.AttnQ_B, 27),
}

LLAMA_CRITICAL_KEYS = {
    "model.embed_tokens.weight": (WeightRole.InEmbed, 0),
    "lm_head.weight": (WeightRole.OutEmbed, 0),
    "model.norm.weight": (WeightRole.OutNorm, 0),
    "model.layers.0.input_layernorm.weight": (WeightRole.AttnNorm, 0),
    "model.layers.0.self_attn.q_proj.weight": (WeightRole.AttnQ_W, 0),
    "model.layers.0.self_attn.k_proj.weight": (WeightRole.AttnK_W, 0),
    "model.layers.0.self_attn.v_proj.weight": (WeightRole.AttnV_W, 0),
    "model.layers.0.self_attn.o_proj.weight": (WeightRole.AttnO_W, 0),
    "model.layers.0.post_attention_layernorm.weight": (WeightRole.MlpNorm, 0),
    "model.layers.0.mlp.gate_proj.weight": (WeightRole.MlpGate_W, 0),
    "model.layers.0.mlp.up_proj.weight": (WeightRole.MlpUp_W, 0),
    "model.layers.0.mlp.down_proj.weight": (WeightRole.MlpDown_W, 0),
}


class TestQwen2WeightMap(unittest.TestCase):
    def test_critical_keys_resolve(self):
        for name, expected in QWEN2_CRITICAL_KEYS.items():
            with self.subTest(name=name):
                self.assertEqual(qwen2_map.resolve_hf_weight(name), expected)

    def test_unknown_key_returns_none(self):
        self.assertIsNone(qwen2_map.resolve_hf_weight("model.layers.0.self_attn.q_norm.weight"))
        self.assertIsNone(qwen2_map.resolve_hf_weight("not.a.weight"))


class TestLlamaWeightMap(unittest.TestCase):
    def test_critical_keys_resolve(self):
        for name, expected in LLAMA_CRITICAL_KEYS.items():
            with self.subTest(name=name):
                self.assertEqual(llama_map.resolve_hf_weight(name), expected)

    def test_bias_keys_ignored(self):
        self.assertIsNone(llama_map.resolve_hf_weight("model.layers.0.self_attn.q_proj.bias"))


if __name__ == "__main__":
    unittest.main()
