from .libllaisys import LIB_LLAISYS, DataType, DeviceType
from .tensor import Tensor
from ctypes import c_bool, c_float, c_size_t


class Ops:
    @staticmethod
    def add(c: Tensor, a: Tensor, b: Tensor):
        LIB_LLAISYS.llaisysAdd(c.lib_tensor(), a.lib_tensor(), b.lib_tensor())

    @staticmethod
    def argmax(max_idx: Tensor, max_val: Tensor, vals: Tensor):
        LIB_LLAISYS.llaisysArgmax(max_idx.lib_tensor(), max_val.lib_tensor(), vals.lib_tensor())

    @staticmethod
    def topk(out_idx: Tensor, out_val: Tensor, vals: Tensor, k: int):
        LIB_LLAISYS.llaisysTopk(out_idx.lib_tensor(), out_val.lib_tensor(), vals.lib_tensor(), c_size_t(k))

    @staticmethod
    def embedding(out: Tensor, index: Tensor, weight: Tensor):
        LIB_LLAISYS.llaisysEmbedding(
            out.lib_tensor(), index.lib_tensor(), weight.lib_tensor()
        )

    @staticmethod
    def linear(out: Tensor, inp: Tensor, weight: Tensor, bias: Tensor):
        LIB_LLAISYS.llaisysLinear(
            out.lib_tensor(),
            inp.lib_tensor(),
            weight.lib_tensor(),
            bias.lib_tensor() if bias is not None else None,
        )

    @staticmethod
    def kv_cache_move(out: Tensor, inp: Tensor, block_ids: Tensor, cut_idx: Tensor, pos_ids: Tensor, max_seq_len: int):
        LIB_LLAISYS.llaisysKvCacheMove(
            out.lib_tensor(),
            inp.lib_tensor(),
            block_ids.lib_tensor(),
            cut_idx.lib_tensor(),
            pos_ids.lib_tensor(),
            c_size_t(max_seq_len),
        )

    @staticmethod
    def rearrange(out: Tensor, inp: Tensor):
        LIB_LLAISYS.llaisysRearrange(out.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def rms_norm(out: Tensor, inp: Tensor, weight: Tensor, eps: float):
        LIB_LLAISYS.llaisysRmsNorm(
            out.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        LIB_LLAISYS.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def paged_attention(attn_val: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_ids, *rest):
        """
        Batched call: (attn_val, q_cat, k_cache, v_cache, block_ids_ll, cut_idx_ll,
                        totlen_ll, max_seq_len:int, scale:float[, is_prefill:bool])

        Defaults is_prefill=True for backward compatibility.
        """
        # Detect calling convention: if rest[0] is a Tensor, it's the batched form
        if len(rest) >= 3 and isinstance(rest[0], Tensor):
            cut_idx = rest[0]
            tot_len = rest[1]
            max_seq_len = rest[2]
            scale = rest[3]
            is_prefill = rest[4] if len(rest) > 4 else True
            LIB_LLAISYS.llaisysPagedAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k_cache.lib_tensor(),
                v_cache.lib_tensor(),
                block_ids.lib_tensor(),
                cut_idx.lib_tensor(),
                tot_len.lib_tensor(),
                c_size_t(max_seq_len),
                c_float(scale),
                c_bool(is_prefill),
            )
        else:
            raise TypeError(
                "paged_attention: legacy per-sample signature is no longer supported. "
                "Use the batched form: (attn_val, q, k_cache, v_cache, block_ids, "
                "cut_idx, tot_len, max_seq_len, scale[, is_prefill])"
            )

    @staticmethod
    def self_attention(attn_val: Tensor, q: Tensor, k: Tensor, v: Tensor, scale: float):
        LIB_LLAISYS.llaisysSelfAttention(
            attn_val.lib_tensor(),
            q.lib_tensor(),
            k.lib_tensor(),
            v.lib_tensor(),
            c_float(scale),
        )

    @staticmethod
    def swiglu(out: Tensor, gate: Tensor, up: Tensor):
        LIB_LLAISYS.llaisysSwiGLU(out.lib_tensor(), gate.lib_tensor(), up.lib_tensor())
