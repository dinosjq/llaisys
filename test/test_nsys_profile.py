import importlib.util
import unittest
from pathlib import Path


def load_nsys_profile():
    path = Path(__file__).parent / "profile" / "nsys_profile.py"
    spec = importlib.util.spec_from_file_location("nsys_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class NsysProfileWarmupTest(unittest.TestCase):
    def test_warmup_prompt_keeps_shape_without_reusing_prefix(self):
        profile = load_nsys_profile()

        warmup_ids = profile.make_warmup_input([7, 8, 9], vocab_size=10)

        self.assertEqual(warmup_ids, [8, 8, 9])


if __name__ == "__main__":
    unittest.main()
