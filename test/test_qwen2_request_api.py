import unittest

from llaisys.libllaisys import LIB_LLAISYS


class ModelRequestApiTest(unittest.TestCase):
    def test_legacy_prompt_keyed_api_is_not_exported(self):
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelInfer"))
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelAbort"))
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2RequestSubmit"))

    def test_request_handle_api_is_exported(self):
        for symbol in (
            "llaisysModelRequestSubmit",
            "llaisysModelRequestAwait",
            "llaisysModelRequestAbort",
            "llaisysModelRequestRelease",
        ):
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(LIB_LLAISYS, symbol))


if __name__ == "__main__":
    unittest.main()
