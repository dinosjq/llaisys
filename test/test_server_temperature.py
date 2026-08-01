import sys
import types
import unittest

from fastapi.testclient import TestClient

import llaisys.server as server


class FakeTokenizer:
    def apply_chat_template(self, *_args, **_kwargs):
        return [1, 2]

    def encode(self, *_args, **_kwargs):
        return [3, 4]

    def decode(self, *_args, **_kwargs):
        return "ok"


class FakeModel:
    temperatures = []

    def __init__(self, *_args, **_kwargs):
        pass

    def close(self):
        pass

    def generate(self, inputs, **kwargs):
        self.temperatures.append(kwargs["temperature"])
        return list(inputs) + [0]


class ServerTemperatureTest(unittest.TestCase):
    def test_explicit_zero_temperature_reaches_both_non_stream_endpoints(self):
        original_qwen2 = server.Qwen2
        original_transformers = sys.modules.get("transformers")
        FakeModel.temperatures = []
        server.Qwen2 = FakeModel
        sys.modules["transformers"] = types.SimpleNamespace(
            AutoTokenizer=types.SimpleNamespace(from_pretrained=lambda *_args, **_kwargs: FakeTokenizer())
        )
        try:
            app = server.create_app("unused")
            with TestClient(app) as client:
                chat = client.post(
                    "/v1/chat/completions",
                    json={"messages": [{"role": "user", "content": "hi"}], "temperature": 0, "stream": False},
                )
                completion = client.post(
                    "/v1/completions",
                    json={"prompt": "hi", "temperature": 0, "stream": False},
                )
            self.assertEqual(chat.status_code, 200)
            self.assertEqual(completion.status_code, 200)
            self.assertEqual(FakeModel.temperatures, [0, 0])
        finally:
            server.Qwen2 = original_qwen2
            if original_transformers is None:
                del sys.modules["transformers"]
            else:
                sys.modules["transformers"] = original_transformers


if __name__ == "__main__":
    unittest.main()
