import ctypes
from ctypes import POINTER, c_char_p, c_int, c_size_t, c_int64, c_float

from ..llaisys_types import llaisysDeviceType_t
from ..tensor import llaisysTensor_t


llaisysModel_t = ctypes.c_void_p
llaisysRequest_t = ctypes.c_void_p


def load_model(lib):
    lib.llaisysModelCreate.argtypes = [
        c_char_p,
        llaisysDeviceType_t,
        POINTER(c_int),
        c_int,
    ]
    lib.llaisysModelCreate.restype = llaisysModel_t

    lib.llaisysModelDestroy.argtypes = [llaisysModel_t]
    lib.llaisysModelDestroy.restype = None

    lib.llaisysModelSetMeta.argtypes = [
        llaisysModel_t,
        POINTER(c_char_p),
        POINTER(ctypes.c_void_p),
        POINTER(c_size_t),
        c_size_t,
    ]
    lib.llaisysModelSetMeta.restype = c_int

    lib.llaisysModelSetWeight.argtypes = [
        llaisysModel_t,
        c_char_p,
        llaisysTensor_t,
    ]
    lib.llaisysModelSetWeight.restype = c_int

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
    "llaisysModel_t",
    "llaisysRequest_t",
    "load_model",
]
