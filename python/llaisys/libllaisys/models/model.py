import ctypes
from ctypes import POINTER, c_int, c_size_t, c_int64, c_float
from enum import IntEnum

from ..llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from ..tensor import llaisysTensor_t


class LlaisysModelArch(IntEnum):
    QWEN2 = 0
    LLAMA = 1


class LlaisysWeightRole(IntEnum):
    IN_EMBED = 0
    OUT_EMBED = 1
    OUT_NORM = 2
    ATTN_NORM = 3
    ATTN_Q_W = 4
    ATTN_Q_B = 5
    ATTN_K_W = 6
    ATTN_K_B = 7
    ATTN_V_W = 8
    ATTN_V_B = 9
    ATTN_O_W = 10
    MLP_NORM = 11
    MLP_GATE_W = 12
    MLP_UP_W = 13
    MLP_DOWN_W = 14


class LlaisysModelMeta(ctypes.Structure):
    _fields_ = [
        ("dtype", llaisysDataType_t),
        ("nlayer", c_size_t),
        ("hs", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("maxseq", c_size_t),
        ("voc", c_size_t),
        ("epsilon", c_float),
        ("theta", c_float),
        ("end_token", c_int64),
    ]


llaisysModel_t = ctypes.c_void_p
llaisysRequest_t = ctypes.c_void_p


def load_model(lib):
    lib.llaisysModelCreate.argtypes = [
        c_int,
        POINTER(LlaisysModelMeta),
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysModelCreate.restype = llaisysModel_t

    lib.llaisysModelDestroy.argtypes = [llaisysModel_t]
    lib.llaisysModelDestroy.restype = None

    lib.llaisysModelSetWeight.argtypes = [
        llaisysModel_t,
        c_int,
        c_size_t,
        llaisysTensor_t,
    ]
    lib.llaisysModelSetWeight.restype = None

    lib.llaisysModelRequestSubmit.argtypes = [
        llaisysModel_t,
        POINTER(c_int64),
        c_size_t,
        c_int64,
        c_int,
        c_float,
        c_float,
    ]
    lib.llaisysModelRequestSubmit.restype = llaisysRequest_t

    lib.llaisysModelRequestAwait.argtypes = [llaisysRequest_t]
    lib.llaisysModelRequestAwait.restype = c_int64
    lib.llaisysModelRequestAbort.argtypes = [llaisysRequest_t]
    lib.llaisysModelRequestAbort.restype = None
    lib.llaisysModelRequestRelease.argtypes = [llaisysRequest_t]
    lib.llaisysModelRequestRelease.restype = None


__all__ = [
    "LlaisysModelArch",
    "LlaisysWeightRole",
    "LlaisysModelMeta",
    "llaisysModel_t",
    "llaisysRequest_t",
    "load_model",
]
