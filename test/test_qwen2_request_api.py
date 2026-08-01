import unittest

from llaisys.libllaisys import LIB_LLAISYS


class Qwen2RequestApiTest(unittest.TestCase):
    def test_legacy_prompt_keyed_api_is_not_exported(self):
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelInfer"))
        self.assertFalse(hasattr(LIB_LLAISYS, "llaisysQwen2ModelAbort"))

    def test_request_handle_api_is_exported(self):
        for symbol in (
            "llaisysQwen2RequestSubmit",
            "llaisysQwen2RequestAwait",
            "llaisysQwen2RequestAbort",
            "llaisysQwen2RequestRelease",
        ):
            with self.subTest(symbol=symbol):
                self.assertTrue(hasattr(LIB_LLAISYS, symbol))


if __name__ == "__main__":
    unittest.main()
