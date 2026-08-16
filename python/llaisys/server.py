"""OpenAI-compatible API server for LLAISYS.

Usage:
    python -m llaisys.server --model /path/to/model --port 8000 --device nvidia
"""
import time
import uuid
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from .schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionChunk,
    Choice,
    ChoiceDelta,
    Usage,
    ModelCard,
    ModelList,
    CompletionRequest,
    Message,
)
from .models import load_causal_lm
from . import DeviceType

logger = logging.getLogger("llaisys.server")

# 全局状态 (lifespan 中初始化)
_state = {
    "model": None,       # Qwen2 / Llama instance
    "tokenizer": None,   # HF tokenizer
    "model_name": "llaisys",
}


def _encode_messages(tokenizer, messages) -> list[int]:
    """Tokenize chat messages using the model's own chat template."""
    conv = [{"role": m.role, "content": m.content} for m in messages]
    try:
        encoded = tokenizer.apply_chat_template(conv, add_generation_prompt=True, tokenize=True)
        if hasattr(encoded, "keys") and "input_ids" in encoded:
            encoded = encoded["input_ids"]
        if hasattr(encoded, "tolist"):
            encoded = encoded.tolist()
        if encoded and isinstance(encoded[0], list):
            encoded = encoded[0]
        return encoded
    except Exception:
        # fallback: generic im_start/im_end template
        prompt_text = ""
        for m in messages:
            prompt_text += f"<|im_start|>{m.role}\n{m.content}<|im_end|>\n"
        prompt_text += "<|im_start|>assistant\n"
        return tokenizer.encode(prompt_text, add_special_tokens=False)


def create_app(model_path: str, device: str = "nvidia", model_name: str | None = None) -> FastAPI:
    """Create the FastAPI app; model loads on startup via lifespan."""
    dev = DeviceType.NVIDIA if device.lower() == "nvidia" else DeviceType.CPU
    _state["model_name"] = model_name or Path(model_path).name

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info("Loading model from %s on %s...", model_path, device)
        from transformers import AutoTokenizer
        _state["tokenizer"] = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        _state["model"] = load_causal_lm(model_path, dev)
        logger.info("Model loaded.")
        yield
        logger.info("Shutting down...")
        if _state["model"] is not None:
            _state["model"].close()
            _state["model"] = None

    app = FastAPI(title="LLAISYS API", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/v1/models", response_model=ModelList)
    async def list_models():
        return ModelList(data=[ModelCard(id=_state["model_name"], owned_by="llaisys")])

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest):
        model = _state["model"]
        tokenizer = _state["tokenizer"]

        input_ids = _encode_messages(tokenizer, req.messages)

        req_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if not req.stream:
            # 非流式: 在线程池中运行阻塞 generate
            tokens = await asyncio.to_thread(
                model.generate,
                input_ids,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature if req.temperature is not None else 0.8,
            )
            generated = tokens[len(input_ids):]
            text = tokenizer.decode(generated, skip_special_tokens=True)

            return ChatCompletionResponse(
                id=req_id,
                created=created,
                model=_state["model_name"],
                choices=[Choice(
                    index=0,
                    message=Message(role="assistant", content=text),
                    finish_reason="stop",
                )],
                usage=Usage(
                    prompt_tokens=len(input_ids),
                    completion_tokens=len(generated),
                    total_tokens=len(tokens),
                ),
            )

        # 流式: SSE
        async def event_stream():
            try:
                async for chunk_data in model.generate_async(
                    input_ids,
                    max_new_tokens=req.max_tokens,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    temperature=req.temperature if req.temperature is not None else 0.8,
                ):
                    token_id = chunk_data["token"]
                    finish = chunk_data.get("finish_reason")
                    text = tokenizer.decode([token_id], skip_special_tokens=True)

                    delta = ChoiceDelta(content=text)
                    if chunk_data["index"] == 0:
                        delta.role = "assistant"

                    chunk = ChatCompletionChunk(
                        id=req_id,
                        created=created,
                        model=_state["model_name"],
                        choices=[Choice(
                            index=0,
                            delta=delta,
                            finish_reason=finish,
                        )],
                    )
                    yield {"data": chunk.model_dump_json(exclude_none=True)}

                yield {"data": "[DONE]"}
            except asyncio.CancelledError:
                # 客户端断开: generate_async 内部已调用 Abort, 这里只记录
                logger.info("Request %s cancelled by client disconnect", req_id)
                raise

        return EventSourceResponse(event_stream())

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest):
        model = _state["model"]
        tokenizer = _state["tokenizer"]

        input_ids = tokenizer.encode(req.prompt, add_special_tokens=False)
        req_id = f"cmpl-{uuid.uuid4().hex[:8]}"
        created = int(time.time())

        if not req.stream:
            tokens = await asyncio.to_thread(
                model.generate,
                input_ids,
                max_new_tokens=req.max_tokens,
                top_k=req.top_k,
                top_p=req.top_p,
                temperature=req.temperature if req.temperature is not None else 0.8,
            )
            generated = tokens[len(input_ids):]
            text = tokenizer.decode(generated, skip_special_tokens=True)

            return JSONResponse(content={
                "id": req_id,
                "object": "text_completion",
                "created": created,
                "model": _state["model_name"],
                "choices": [{
                    "index": 0,
                    "text": text,
                    "finish_reason": "stop",
                }],
                "usage": {
                    "prompt_tokens": len(input_ids),
                    "completion_tokens": len(generated),
                    "total_tokens": len(tokens),
                },
            })

        async def event_stream():
            import json as _json
            try:
                async for chunk_data in model.generate_async(
                    input_ids,
                    max_new_tokens=req.max_tokens,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    temperature=req.temperature if req.temperature is not None else 0.8,
                ):
                    token_id = chunk_data["token"]
                    finish = chunk_data.get("finish_reason")
                    text = tokenizer.decode([token_id], skip_special_tokens=True)
                    yield {"data": _json.dumps({
                        "id": req_id,
                        "object": "text_completion",
                        "created": created,
                        "model": _state["model_name"],
                        "choices": [{
                            "index": 0,
                            "text": text,
                            "finish_reason": finish,
                        }],
                    })}
                yield {"data": "[DONE]"}
            except asyncio.CancelledError:
                logger.info("Request %s cancelled by client disconnect", req_id)
                raise

        return EventSourceResponse(event_stream())

    return app


def main():
    """Entry point for `python -m llaisys.server`."""
    import argparse
    parser = argparse.ArgumentParser(description="LLAISYS OpenAI-compatible API Server")
    parser.add_argument("--model", type=str, required=True, help="Path to model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", type=str, default="nvidia", choices=["nvidia", "cpu"])
    parser.add_argument("--model-name", type=str, default=None,
                        help="Public model id (default: basename of --model)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    import uvicorn
    app = create_app(args.model, args.device, args.model_name)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
