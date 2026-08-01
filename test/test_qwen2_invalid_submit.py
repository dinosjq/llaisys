import os
import unittest
from ctypes import POINTER, c_float, c_int, c_int64, c_size_t

import llaisys
from llaisys.libllaisys import LIB_LLAISYS
from llaisys.models.qwen2 import Qwen2


class InvalidSubmitTest(unittest.TestCase):
    def test_zero_length_request_returns_null_without_crashing(self):
        model_path = os.environ.get("LLAISYS_TEST_MODEL", "/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B")
        model = Qwen2(model_path, llaisys.DeviceType.NVIDIA)
        try:
            request = LIB_LLAISYS.llaisysQwen2RequestSubmit(
                model._model,
                POINTER(c_int64)(),
                c_size_t(0),
                c_int64(1),
                c_int(1),
                c_float(1.0),
                c_float(1.0),
            )
            self.assertFalse(request)
        finally:
            model.close()


if __name__ == "__main__":
    unittest.main()
