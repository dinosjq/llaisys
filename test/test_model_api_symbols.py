import unittest

from llaisys.libllaisys import LIB_LLAISYS


class ModelApiSymbolsTest(unittest.TestCase):
    def test_unified_symbols_exist(self):
        for name in (
            "llaisysModelCreate",
            "llaisysModelDestroy",
            "llaisysModelSetWeight",
            "llaisysModelRequestSubmit",
            "llaisysModelRequestAwait",
            "llaisysModelRequestAbort",
            "llaisysModelRequestRelease",
        ):
            self.assertTrue(hasattr(LIB_LLAISYS, name), name)

    def test_old_qwen2_symbols_gone(self):
        for name in (
            "llaisysQwen2ModelCreate",
            "llaisysQwen2ModelWeights",
            "llaisysQwen2RequestSubmit",
        ):
            self.assertFalse(hasattr(LIB_LLAISYS, name), name)


if __name__ == "__main__":
    unittest.main()
