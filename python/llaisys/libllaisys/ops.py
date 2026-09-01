from .tensor import llaisysTensor_t
from ctypes import c_bool, c_float, c_int, c_size_t, c_void_p

def load_ops(lib):
    lib.llaisysAdd.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysAdd.restype = None

    lib.llaisysArgmax.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysArgmax.restype = None

    lib.llaisysTopk.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_size_t]
    lib.llaisysTopk.restype = None

    lib.llaisysEmbedding.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysEmbedding.restype = None

    lib.llaisysLinear.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysLinear.restype = None

    lib.llaisysDequantW8.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysDequantW8.restype = None

    lib.llaisysQuantizeW8.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysQuantizeW8.restype = None

    lib.llaisysQuantizeAct.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysQuantizeAct.restype = None

    lib.llaisysLinearW8A8.argtypes = [
        llaisysTensor_t,  # out
        llaisysTensor_t,  # in_q (已量化 I8)
        llaisysTensor_t,  # a_scale (per-token F32)
        llaisysTensor_t,  # weight_i8
        llaisysTensor_t,  # w_scale
        llaisysTensor_t,  # bias (可 None)
    ]
    lib.llaisysLinearW8A8.restype = None

    lib.llaisysKvCacheMove.argtypes = [
        llaisysTensor_t,  # out
        llaisysTensor_t,  # in
        llaisysTensor_t,  # block_ids_tensor
        llaisysTensor_t,  # cut_idx_tensor
        llaisysTensor_t,  # pos_ids_tensor
        llaisysTensor_t,  # scale_tensor (量化写入时输出 per-(token,nkvh) scale，可 None)
    ]
    lib.llaisysKvCacheMove.restype = None

    lib.llaisysRearrange.argtypes = [llaisysTensor_t, llaisysTensor_t]
    lib.llaisysRearrange.restype = None

    lib.llaisysRmsNorm.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysRmsNorm.restype = None

    lib.llaisysRmsNormQuant.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysRmsNormQuant.restype = None

    lib.llaisysROPE.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, c_float]
    lib.llaisysROPE.restype = None
    lib.llaisysROPEInvFreq.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysROPEInvFreq.restype = None

    lib.llaisysPagedAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k_cache
        llaisysTensor_t,  # v_cache
        llaisysTensor_t,  # block_ids
        llaisysTensor_t,  # cut_idx
        llaisysTensor_t,  # tot_len
        c_size_t,         # tot_task_num
        c_size_t,         # max_seq_len
        c_float,          # scale
        c_bool,           # is_prefill
        llaisysTensor_t,  # attn_acc (flash_decoding ctx buf)
        llaisysTensor_t,  # attn_sum (flash_decoding ctx buf)
        llaisysTensor_t,  # attn_max (flash_decoding ctx buf)
        llaisysTensor_t,  # k_scale (量化 KV cache 的 per-token scale，可 None)
        llaisysTensor_t,  # v_scale (量化 KV cache 的 per-token scale，可 None)
    ]
    lib.llaisysPagedAttention.restype = None

    lib.llaisysSelfAttention.argtypes = [
        llaisysTensor_t,  # attn_val
        llaisysTensor_t,  # q
        llaisysTensor_t,  # k
        llaisysTensor_t,  # v
        c_float    # scale
    ]
    lib.llaisysSelfAttention.restype = None

    lib.llaisysSwiGLU.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysSwiGLU.restype = None

    lib.llaisysSwiGLUQuant.argtypes = [llaisysTensor_t, llaisysTensor_t, llaisysTensor_t, llaisysTensor_t]
    lib.llaisysSwiGLUQuant.restype = None
