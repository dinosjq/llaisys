"""Smoke: Python config flatten helper used by model loaders."""

import unittest

from llaisys.models.meta_utils import flatten_config, resolve_eos_token


class MetaFlattenTest(unittest.TestCase):
    def test_flatten_scalars_and_nested(self):
        cfg = {
            "num_hidden_layers": 2,
            "hidden_size": 64,
            "rope_scaling": {"rope_type": "llama3", "factor": 8.0},
            "eos_token_id": [1, 2],
        }
        flat = flatten_config(cfg, eos_token_id=resolve_eos_token(cfg, pick_last=True))
        self.assertEqual(flat["num_hidden_layers"], 2)
        self.assertEqual(flat["rope_scaling.rope_type"], "llama3")
        self.assertEqual(flat["rope_scaling.factor"], 8.0)
        self.assertEqual(flat["eos_token_id"], 2)
        self.assertNotIn("eos_token_ids", flat)


if __name__ == "__main__":
    unittest.main()
