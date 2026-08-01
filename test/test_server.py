"""Integration tests for the OpenAI-compatible API server.

Starts a real uvicorn server subprocess (model load takes ~30s),
then exercises the OpenAI-compatible endpoints over HTTP.

Usage:
    python test/test_server.py --model <model_path> [--device nvidia] [--port 8321]
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

DEFAULT_MODEL = "/home/songjq/models/DeepSeek-R1-Distill-Qwen-1.5B"


def wait_ready(base: str, timeout: float = 120.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(f"{base}/health", timeout=2.0)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(1.0)
    raise RuntimeError("server did not become ready in time")


def test_health(base):
    r = httpx.get(f"{base}/health")
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}
    print("[PASS] health")


def test_list_models(base):
    r = httpx.get(f"{base}/v1/models")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
    print("[PASS] list_models")


def test_chat_non_streaming(base):
    r = httpx.post(f"{base}/v1/chat/completions", json={
        "model": "qwen2-1.5b",
        "messages": [{"role": "user", "content": "1+1="}],
        "max_tokens": 16,
        "stream": False,
    }, timeout=120.0)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "chat.completion"
    assert len(data["choices"]) == 1
    assert data["choices"][0]["message"]["role"] == "assistant"
    assert data["choices"][0]["finish_reason"] is not None
    assert data["usage"]["prompt_tokens"] > 0
    print("[PASS] chat_non_streaming:", repr(data["choices"][0]["message"]["content"][:60]))


def test_chat_streaming(base):
    chunks = []
    done = False
    with httpx.stream("POST", f"{base}/v1/chat/completions", json={
        "model": "qwen2-1.5b",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 8,
        "stream": True,
    }, timeout=120.0) as r:
        assert r.status_code == 200
        for line in r.iter_lines():
            if not line.startswith("data:"):
                continue
            data_str = line[5:].strip()
            if data_str == "[DONE]":
                done = True
                break
            chunks.append(json.loads(data_str))

    assert done, "missing [DONE] terminator"
    assert len(chunks) > 0, "no chunks received"
    assert chunks[0]["object"] == "chat.completion.chunk"
    assert chunks[0]["choices"][0]["delta"].get("role") == "assistant"
    print(f"[PASS] chat_streaming: {len(chunks)} chunks + [DONE]")


def test_completions_non_stream(base):
    r = httpx.post(f"{base}/v1/completions", json={
        "model": "qwen2-1.5b",
        "prompt": "The capital of France is",
        "max_tokens": 8,
        "stream": False,
    }, timeout=120.0)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["object"] == "text_completion"
    assert len(data["choices"]) > 0
    print("[PASS] completions_non_stream:", repr(data["choices"][0]["text"][:60]))


def test_cancellation(base):
    """Client disconnect mid-stream must not crash the server."""
    with httpx.stream("POST", f"{base}/v1/chat/completions", json={
        "model": "qwen2-1.5b",
        "messages": [{"role": "user", "content": "Write a very long essay about history"}],
        "max_tokens": 512,
        "stream": True,
    }, timeout=120.0) as r:
        for line in r.iter_lines():
            if line.startswith("data:"):
                break  # disconnect after first chunk

    # server must still be alive and serving
    time.sleep(1.0)
    r = httpx.get(f"{base}/health", timeout=5.0)
    assert r.status_code == 200
    # and able to serve a new request after the abort
    r = httpx.post(f"{base}/v1/chat/completions", json={
        "model": "qwen2-1.5b",
        "messages": [{"role": "user", "content": "2+2="}],
        "max_tokens": 8,
        "stream": False,
    }, timeout=120.0)
    assert r.status_code == 200
    print("[PASS] cancellation: server alive and serving after client disconnect")

def test_identical_prompt_request_isolation(base):
    payload = {
        "model": "qwen2-1.5b",
        "prompt": "A request handle must isolate this prompt.",
        "max_tokens": 8,
        "stream": False,
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        responses = list(pool.map(lambda _: httpx.post(f"{base}/v1/completions", json=payload, timeout=120.0), range(2)))
    assert all(response.status_code == 200 for response in responses)
    assert all(response.json()["choices"] for response in responses)
    print("[PASS] identical prompts: independent concurrent completions")

def test_identical_prompt_cancellation_isolation(base):
    payload = {
        "model": "qwen2-1.5b",
        "prompt": "Keep this identical request active while its peer is cancelled.",
        "max_tokens": 32,
        "stream": True,
    }
    ready = threading.Barrier(2)

    def cancel_one():
        with httpx.stream("POST", f"{base}/v1/completions", json=payload, timeout=120.0) as response:
            assert response.status_code == 200
            ready.wait(timeout=30)
            for line in response.iter_lines():
                if line.startswith("data:"):
                    return

    def complete_other():
        done = False
        with httpx.stream("POST", f"{base}/v1/completions", json=payload, timeout=120.0) as response:
            assert response.status_code == 200
            ready.wait(timeout=30)
            for line in response.iter_lines():
                if line == "data: [DONE]":
                    done = True
                    break
        return done

    with ThreadPoolExecutor(max_workers=2) as pool:
        cancelled = pool.submit(cancel_one)
        completed = pool.submit(complete_other)
        cancelled.result()
        assert completed.result(), "the uncancelled identical request did not complete"
    assert httpx.get(f"{base}/health", timeout=5.0).status_code == 200
    print("[PASS] identical prompts: cancelling one does not affect the other")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--device", type=str, default="nvidia")
    parser.add_argument("--port", type=int, default=8321)
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    proc = subprocess.Popen(
        [sys.executable, "-m", "llaisys.server",
         "--model", args.model, "--device", args.device,
         "--host", "127.0.0.1", "--port", str(args.port)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_ready(base)
        test_health(base)
        test_list_models(base)
        test_chat_non_streaming(base)
        test_chat_streaming(base)
        test_completions_non_stream(base)
        test_cancellation(base)
        test_identical_prompt_request_isolation(base)
        test_identical_prompt_cancellation_isolation(base)
        print("\nAll server tests passed.")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == "__main__":
    main()
