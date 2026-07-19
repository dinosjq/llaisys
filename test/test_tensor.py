import llaisys

import torch
from test_utils import *
import argparse
import subprocess
import sys


def test_tensor():
    torch_tensor = torch.arange(60, dtype=torch_dtype("i64")).reshape(3, 4, 5)
    llaisys_tensor = llaisys.Tensor(
        (3, 4, 5), dtype=llaisys_dtype("i64"), device=llaisys_device("cpu")
    )

    # Test load
    print("===Test load===")
    llaisys_tensor.load(torch_tensor.data_ptr())
    llaisys_tensor.debug()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor, torch_tensor)

    # Test view
    print("===Test view===")
    torch_tensor_view = torch_tensor.view(6, 10)
    llaisys_tensor_view = llaisys_tensor.view(6, 10)
    llaisys_tensor_view.debug()
    assert llaisys_tensor_view.shape() == torch_tensor_view.shape
    assert llaisys_tensor_view.strides() == torch_tensor_view.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_view, torch_tensor_view)

    # Test permute
    print("===Test permute===")
    torch_tensor_perm = torch_tensor.permute(2, 0, 1)
    llaisys_tensor_perm = llaisys_tensor.permute(2, 0, 1)
    llaisys_tensor_perm.debug()
    assert llaisys_tensor_perm.shape() == torch_tensor_perm.shape
    assert llaisys_tensor_perm.strides() == torch_tensor_perm.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_perm, torch_tensor_perm)

    # Test slice
    print("===Test slice===")
    torch_tensor_slice = torch_tensor[:, :, 1:4]
    llaisys_tensor_slice = llaisys_tensor.slice(2, 1, 4)
    llaisys_tensor_slice.debug()
    assert llaisys_tensor_slice.shape() == torch_tensor_slice.shape
    assert llaisys_tensor_slice.strides() == torch_tensor_slice.stride()
    assert llaisys_tensor.is_contiguous() == torch_tensor.is_contiguous()
    assert check_equal(llaisys_tensor_slice, torch_tensor_slice)


# ============ Generalized view tests (2026-07-18) ============

def test_view_column_slice_split():
    """T1: Column slice of fused buffer → split dim (motivating case)."""
    print("===Test view: column slice → split===")
    t = torch.arange(4096, dtype=torch.float32).reshape(2, 2048)
    lt = llaisys.Tensor((2, 2048), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
    lt.load(t.data_ptr())

    # Column slice → non-contiguous (row gap from interleaved k/v)
    t_slice = t[:, :1536]
    lt_slice = lt.slice(1, 0, 1536)
    assert not lt_slice.is_contiguous(), "column slice should be non-contiguous"

    # Split inner dim: [tot, 1536] → [tot, 12, 128]
    t_view = t_slice.view(2, 12, 128)
    lt_view = lt_slice.view(2, 12, 128)
    assert list(lt_view.shape()) == list(t_view.shape), f"shape: {lt_view.shape()} vs {t_view.shape}"
    # Strides match PyTorch exactly for the split case
    assert list(lt_view.strides()) == list(t_view.stride()), \
        f"strides: {lt_view.strides()} vs {t_view.stride()}"
    assert check_equal(lt_view, t_view)
    print("[PASS] view: column slice → split")


def test_view_permute_merge():
    """T2: Permuted tensor → merge dims (non-contiguous but view-compatible).

    Uses int32 to avoid potential bf16/fp16 confusion and simplify debugging.
    """
    print("===Test view: permute → merge===")

    # Create and load as int32 for unambiguous element-size / value inspection
    t_base = torch.arange(60, dtype=torch.int32).reshape(3, 4, 5)
    t = t_base.permute(2, 0, 1)  # shape [5,3,4], strides [1,20,5] — NOT contiguous

    lt = llaisys.Tensor((3, 4, 5), dtype=llaisys.DataType.I32, device=llaisys.DeviceType.CPU)
    lt.load(t_base.data_ptr())
    lt = lt.permute(2, 0, 1)
    assert not lt.is_contiguous(), "permuted tensor should be non-contiguous"

    # PyTorch allows view(5,12) — merge dims 1 and 2 (20 == 4*5)
    t_view = t.view(5, 12)
    lt_view = lt.view(5, 12)
    assert list(lt_view.shape()) == [5, 12]
    assert list(lt_view.strides()) == [1, 5], f"expected [1,5], got {lt_view.strides()}"
    assert check_equal(lt_view, t_view)
    print("[PASS] view: permute → merge")


def test_view_incompatible_column_flatten():
    """T3: Column-sliced tensor (tot>1) cannot flatten — must throw.

    Run in subprocess because the C bridge lacks try/catch and a C++ exception
    through extern "C" into ctypes terminates the Python interpreter.
    """
    print("===Test view: incompatible column flatten (throw)===")
    script = r"""
import llaisys, numpy as np
t = llaisys.Tensor((2, 2048), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
t.load(np.arange(2*2048, dtype=np.float32).ctypes.data)
ts = t.slice(1, 0, 1536)
# Python wrapper has no numel(); hardcode total=2*1536=3072
ts.view(3072)
# If we get here, view succeeded when it should have thrown
print("SHOULD_NOT_REACH")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    # Incompatible view throws std::runtime_error → C bridge lack of try/catch →
    # std::terminate → SIGABRT / non-zero exit.
    assert "SHOULD_NOT_REACH" not in result.stdout, \
        f"incompatible view should have thrown, but succeeded. stdout: {result.stdout}"
    print("[PASS] view: incompatible column flatten throws")


def test_view_single_row_flatten():
    """T4: Single-row column slice can flatten (tot==1, one chunk)."""
    print("===Test view: single row column slice flatten===")
    t = torch.arange(2048, dtype=torch.float32).reshape(1, 2048)
    lt = llaisys.Tensor((1, 2048), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
    lt.load(t.data_ptr())

    lt_slice = lt.slice(1, 0, 1536)
    t_slice = t[:, :1536]
    # tot==1 → only one row, row-gap doesn't matter → flatten is valid
    t_view = t_slice.reshape(1536)  # PyTorch reshape (view succeeds for this case)
    lt_view = lt_slice.view(1536)
    assert list(lt_view.shape()) == [1536]
    assert check_equal(lt_view, t_view)
    print("[PASS] view: single row column slice → flatten")


def test_view_size1_insertion():
    """T5: Size-1 dimension handling — contiguous and non-contiguous paths."""
    print("===Test view: size-1 dim handling===")

    # Part A: Contiguous path (fast-path regression)
    t = torch.arange(24, dtype=torch.float32).reshape(24)
    lt = llaisys.Tensor((24,), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
    lt.load(t.data_ptr())
    for t_view, lt_view in [
        (t.view(1, 24), lt.view(1, 24)),
        (t.view(24, 1), lt.view(24, 1)),
        (t.view(1, 24, 1, 1), lt.view(1, 24, 1, 1)),
    ]:
        assert check_equal(lt_view, t_view)
    # Remove size-1: view(1,24,1).view(24) — contiguous path for intermediate
    assert check_equal(lt.view(1, 24, 1).view(24), t.view(1, 24, 1).view(24))

    # Part B: Non-contiguous path — size-1 old dim skip in generalized view
    # Create [2,2,4] contiguous, slice dim-1 to size-1 → [2,1,4] strides [8,4,1]
    # NOT contiguous (dim 0 stride 8 ≠ 4*1 = 4). Then view(2,4) — must work.
    t2 = torch.arange(16, dtype=torch.float32).reshape(2, 2, 4)
    lt2 = llaisys.Tensor((2, 2, 4), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
    lt2.load(t2.data_ptr())
    t2s = t2[:, 1:2, :]  # shape [2,1,4], strides [8,4,1]
    lt2s = lt2.slice(1, 1, 2)
    assert not lt2s.is_contiguous(), "sliced tensor should be non-contiguous"
    # view(2,4) — size-1 old dim (dim 1) must not create chunk boundary
    t2v = t2s.view(2, 4)
    lt2v = lt2s.view(2, 4)
    assert list(lt2v.shape()) == [2, 4]
    assert check_equal(lt2v, t2v)

    # Part C: Leading size-1 new dims on non-contiguous tensor
    # [1,1536] column-sliced → view(1,1,1536) → leading size-1 → generalized path
    t3 = torch.arange(2048, dtype=torch.float32).reshape(1, 2048)
    lt3 = llaisys.Tensor((1, 2048), dtype=llaisys.DataType.F32, device=llaisys.DeviceType.CPU)
    lt3.load(t3.data_ptr())
    lt3s = lt3.slice(1, 0, 1536)
    t3s = t3[:, :1536]
    lt3v = lt3s.view(1, 1, 1536)
    t3v = t3s.view(1, 1, 1536)
    assert check_equal(lt3v, t3v)
    print("[PASS] view: size-1 dim handling")


if __name__ == "__main__":
    import sys as _sys, subprocess as _sp, os as _os, argparse as _ap

    p = _ap.ArgumentParser()
    p.add_argument("--test", type=str, default="all",
                   choices=["all","regression","T1","T2","T3","T4","T5"])
    a = p.parse_args()

    test_map = {
        "regression": ["test_tensor"],
        "T1": ["test_view_column_slice_split"],
        "T2": ["test_view_permute_merge"],
        "T3": ["test_view_incompatible_column_flatten"],
        "T4": ["test_view_single_row_flatten"],
        "T5": ["test_view_size1_insertion"],
    }

    if a.test == "all":
        for label in ["T1","T2","T3","T4","T5","regression"]:
            print(f"--- Running {label} (isolated subprocess) ---")
            r = _sp.run([_sys.executable, __file__, "--test", label],
                        capture_output=True, text=True,
                        cwd=_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
                        timeout=30)
            if r.returncode != 0:
                print(f"FAIL: {label}")
                print(r.stdout[-300:] if r.stdout else "")
                print(r.stderr[-300:] if r.stderr else "")
                _sys.exit(1)
            print(f"  PASS")
        print("\n\033[92mAll tensor tests passed!\033[0m\n")
    else:
        for fn_name in test_map[a.test]:
            globals()[fn_name]()
