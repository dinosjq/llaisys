import asyncio
import threading
import types
import unittest

from llaisys.models.qwen2 import Qwen2
import llaisys.models.qwen2 as qwen2_module


class BlockingRequestLibrary:
    def __init__(self):
        self.await_started = threading.Event()
        self.allow_await_return = threading.Event()
        self.await_finished = threading.Event()
        self.released = threading.Event()

    def llaisysQwen2RequestAwait(self, _request):
        self.await_started.set()
        self.allow_await_return.wait()
        self.await_finished.set()
        return -2

    def llaisysQwen2RequestAbort(self, _request):
        return None

    def llaisysQwen2RequestRelease(self, _request):
        self.released.set()


class RequestLifetimeTest(unittest.IsolatedAsyncioTestCase):
    async def test_cancellation_defers_release_until_blocking_await_finishes(self):
        fake_lib = BlockingRequestLibrary()
        model = Qwen2.__new__(Qwen2)
        model._meta = types.SimpleNamespace(end_token=0)
        model._submit_request = lambda *_args: object()
        original_lib = qwen2_module.LIB_LLAISYS
        qwen2_module.LIB_LLAISYS = fake_lib
        try:
            stream = model.generate_async([1], max_new_tokens=1)
            next_chunk = asyncio.create_task(anext(stream))
            await asyncio.wait_for(asyncio.to_thread(fake_lib.await_started.wait), 1)
            next_chunk.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await next_chunk
            released_before_await_finished = fake_lib.released.is_set()
            fake_lib.allow_await_return.set()
            await asyncio.wait_for(asyncio.to_thread(fake_lib.await_finished.wait), 1)
            await asyncio.wait_for(asyncio.to_thread(fake_lib.released.wait), 1)
            self.assertFalse(released_before_await_finished)
        finally:
            fake_lib.allow_await_return.set()
            qwen2_module.LIB_LLAISYS = original_lib


if __name__ == "__main__":
    unittest.main()
