"""Helpers: flatten HF config.json → typed byte map for llaisysModelSetMeta."""

from __future__ import annotations

import struct
from ctypes import POINTER, c_char, c_char_p, c_size_t, c_void_p, cast
from typing import Any, Dict, List, Mapping, MutableMapping, Optional


def _flatten_value(prefix: str, value: Any, out: MutableMapping[str, Any]) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            _flatten_value(key, v, out)
        return
    if isinstance(value, bool):
        # Keep bools as int so C++ can read int64_t if needed; meta paths use strings/floats/ints.
        out[prefix] = int(value)
        return
    if value is None:
        return
    if isinstance(value, (list, tuple)):
        # Nested lists are not passed; callers should pre-resolve eos lists.
        return
    out[prefix] = value


def flatten_config(config: Mapping[str, Any], *, eos_token_id: Optional[int] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in config.items():
        if k in ("eos_token_id", "eos_token_ids"):
            continue
        _flatten_value(str(k), v, out)
    if eos_token_id is not None:
        out["eos_token_id"] = int(eos_token_id)
    return out


def _pack_value(value: Any) -> bytes:
    if isinstance(value, bool):
        return struct.pack("<q", int(value))
    if isinstance(value, int):
        return struct.pack("<q", int(value))
    if isinstance(value, float):
        return struct.pack("<f", float(value))
    if isinstance(value, str):
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    raise TypeError(f"unsupported meta value type: {type(value)!r}")


def set_model_meta(lib, model, meta: Mapping[str, Any]) -> None:
    keys = list(meta.keys())
    packed: List[bytes] = [_pack_value(meta[k]) for k in keys]
    # Keep buffers alive across the C call (raw bytes, not NUL-terminated strings).
    bufs = [(c_char * len(blob)).from_buffer_copy(blob) for blob in packed]
    key_arr = (c_char_p * len(keys))(*[k.encode("utf-8") for k in keys])
    val_arr = (c_void_p * len(bufs))(*[cast(b, c_void_p) for b in bufs])
    nbytes_arr = (c_size_t * len(packed))(*[len(blob) for blob in packed])
    rc = int(lib.llaisysModelSetMeta(model, key_arr, val_arr, nbytes_arr, c_size_t(len(keys))))
    if rc != 0:
        raise RuntimeError("llaisysModelSetMeta failed")


def resolve_eos_token(config: Mapping[str, Any], *, pick_last: bool = False) -> int:
    eos = config.get("eos_token_id", config.get("eos_token_ids", None))
    if isinstance(eos, list):
        if not eos:
            return 0
        return int(eos[-1] if pick_last else eos[0])
    if eos is None:
        return 0
    return int(eos)


class SimpleMeta:
    def __init__(self, end_token: int):
        self.end_token = int(end_token)
