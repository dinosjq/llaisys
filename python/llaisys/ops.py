from .libllaisys import LIB_LLAISYS, DataType, DeviceType
from .tensor import Tensor
from ctypes import c_bool, c_float, c_int, c_size_t


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
    def dequant_w8(out: Tensor, weight_i8: Tensor, scale: Tensor):
        LIB_LLAISYS.llaisysDequantW8(out.lib_tensor(), weight_i8.lib_tensor(), scale.lib_tensor())

    @staticmethod
    def quantize_w8(weight_i8: Tensor, scale: Tensor, weight_fp: Tensor):
        LIB_LLAISYS.llaisysQuantizeW8(weight_i8.lib_tensor(), scale.lib_tensor(), weight_fp.lib_tensor())

    @staticmethod
    def quantize_act(out_q: Tensor, out_scale: Tensor, inp: Tensor):
        """激活 per-token 量化（独立步骤）：in → out_q(I8) + out_scale(F32)。"""
        LIB_LLAISYS.llaisysQuantizeAct(out_q.lib_tensor(), out_scale.lib_tensor(), inp.lib_tensor())

    @staticmethod
    def linear_w8a8(out: Tensor, in_q: Tensor, a_scale: Tensor, weight_i8: Tensor, w_scale: Tensor,
                    bias: Tensor = None):
        """W8A8 量化线性：输入已量化（in_q I8 + a_scale per-token），权重 int8 + per-channel scale。"""
        LIB_LLAISYS.llaisysLinearW8A8(
            out.lib_tensor(),
            in_q.lib_tensor(),
            a_scale.lib_tensor(),
            weight_i8.lib_tensor(),
            w_scale.lib_tensor(),
            bias.lib_tensor() if bias is not None else None,
        )

    @staticmethod
    def kv_cache_move(out: Tensor, inp: Tensor, block_ids: Tensor, cut_idx: Tensor, pos_ids: Tensor, scale: Tensor = None):
        """out 为 I8 时走量化写入，scale 输出 per-(token,nkvh) F32 scale（必须提供）。"""
        LIB_LLAISYS.llaisysKvCacheMove(
            out.lib_tensor(),
            inp.lib_tensor(),
            block_ids.lib_tensor(),
            cut_idx.lib_tensor(),
            pos_ids.lib_tensor(),
            scale.lib_tensor() if scale is not None else None,
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
    def rms_norm_quant(out_q: Tensor, out_scale: Tensor, inp: Tensor, weight: Tensor, eps: float):
        """融合 rms_norm + Hadamard + per-token 量化：in → out_q(I8) + out_scale(F32)。"""
        LIB_LLAISYS.llaisysRmsNormQuant(
            out_q.lib_tensor(), out_scale.lib_tensor(), inp.lib_tensor(), weight.lib_tensor(), c_float(eps)
        )

    @staticmethod
    def rope(out: Tensor, inp: Tensor, pos_ids: Tensor, theta: float):
        LIB_LLAISYS.llaisysROPE(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), c_float(theta)
        )

    @staticmethod
    def rope_inv_freq(out: Tensor, inp: Tensor, pos_ids: Tensor, inv_freq: Tensor):
        LIB_LLAISYS.llaisysROPEInvFreq(
            out.lib_tensor(), inp.lib_tensor(), pos_ids.lib_tensor(), inv_freq.lib_tensor()
        )

    @staticmethod
    def paged_attention(attn_val: Tensor, q: Tensor, k_cache: Tensor, v_cache: Tensor, block_ids, *rest):
        """
        Batched call: (attn_val, q_cat, k_cache, v_cache, block_ids_ll, cut_idx_ll,
                        totlen_ll, tot_task_num:int, scale:float
                        [, is_prefill:bool, attn_acc, attn_sum, attn_max])

        tot_task_num: prefill = Σceil(seq_len/BLOCK_M)；decode = Σ块数（flash_decoding 的 grid.x）。
        Defaults is_prefill=True. attn_acc/sum/max are optional buffer tensors
        for flash_decoding (required when is_prefill=False). k_scale/v_scale are
        optional per-token scales for quantized (I8) KV cache.
        """
        if len(rest) >= 4 and isinstance(rest[0], Tensor):
            cut_idx = rest[0]
            tot_len = rest[1]
            tot_task_num = rest[2]
            # 新签名含 max_seq_len：rest[3] 为 int；旧签名（无 max_seq_len）rest[3] 为 scale(float)，max_seq_len 默认 0
            if len(rest) > 3 and isinstance(rest[3], int) and not isinstance(rest[3], bool):
                max_seq_len = rest[3]
                scale = rest[4]
                buf_start = 5
            else:
                max_seq_len = 0
                scale = rest[3]
                buf_start = 4
            # is_prefill: rest[buf_start] if present and not a Tensor, else True
            if len(rest) > buf_start and not isinstance(rest[buf_start], Tensor):
                is_prefill = rest[buf_start]
                buf_start += 1
            else:
                is_prefill = True
            # Optional flash_decoding context buffers
            attn_acc = rest[buf_start] if len(rest) > buf_start else None
            attn_sum = rest[buf_start + 1] if len(rest) > buf_start + 1 else None
            attn_max = rest[buf_start + 2] if len(rest) > buf_start + 2 else None
            # Optional quantized KV cache scales
            k_scale = rest[buf_start + 3] if len(rest) > buf_start + 3 else None
            v_scale = rest[buf_start + 4] if len(rest) > buf_start + 4 else None
            LIB_LLAISYS.llaisysPagedAttention(
                attn_val.lib_tensor(),
                q.lib_tensor(),
                k_cache.lib_tensor(),
                v_cache.lib_tensor(),
                block_ids.lib_tensor(),
                cut_idx.lib_tensor(),
                tot_len.lib_tensor(),
                c_size_t(tot_task_num),
                c_size_t(max_seq_len),
                c_float(scale),
                c_bool(is_prefill),
                attn_acc.lib_tensor() if attn_acc else None,
                attn_sum.lib_tensor() if attn_sum else None,
                attn_max.lib_tensor() if attn_max else None,
                k_scale.lib_tensor() if k_scale else None,
                v_scale.lib_tensor() if v_scale else None,
            )
        else:
            raise TypeError(
                "paged_attention: legacy per-sample signature is no longer supported. "
                "Use the batched form: (attn_val, q, k_cache, v_cache, block_ids, "
                "cut_idx, tot_len, tot_task_num, scale[, is_prefill])"
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

    @staticmethod
    def swiglu_quant(out_q: Tensor, out_scale: Tensor, gate: Tensor, up: Tensor):
        """融合 swiglu + Hadamard + per-token 量化：gate/up → out_q(I8) + out_scale(F32)。"""
        LIB_LLAISYS.llaisysSwiGLUQuant(
            out_q.lib_tensor(), out_scale.lib_tensor(), gate.lib_tensor(), up.lib_tensor()
        )
