"""OpenAI-compatible request/response schemas."""
from pydantic import BaseModel, Field
from typing import Optional, Literal


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen2-1.5b"
    messages: list[Message]
    max_tokens: Optional[int] = Field(default=128, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=10, ge=1, le=100)
    stream: bool = False
    # logprobs / top_logprobs: deferred to next iteration
    stop: Optional[list[str]] = None
    n: int = 1

    class Config:
        extra = "allow"


class ChoiceDelta(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None


class Choice(BaseModel):
    index: int = 0
    message: Optional[Message] = None
    delta: Optional[ChoiceDelta] = None
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str = "chatcmpl-0"
    object: str = "chat.completion"
    created: int = 0
    model: str = "qwen2-1.5b"
    choices: list[Choice]
    usage: Optional[Usage] = None


class ChatCompletionChunk(BaseModel):
    id: str = "chatcmpl-0"
    object: str = "chat.completion.chunk"
    created: int = 0
    model: str = "qwen2-1.5b"
    choices: list[Choice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = 0
    owned_by: str = "llaisys"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]


class CompletionRequest(BaseModel):
    model: str = "qwen2-1.5b"
    prompt: str
    max_tokens: Optional[int] = Field(default=128, ge=1, le=8192)
    temperature: Optional[float] = Field(default=0.8, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=10, ge=1, le=100)
    stream: bool = False
    # logprobs: deferred to next iteration

    class Config:
        extra = "allow"
