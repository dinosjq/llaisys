import sys
import os
from pathlib import Path

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from test_utils import random_tensor, check_equal, benchmark


def torch_linear(out, x, w, bias):
    torch.nn.functional.linear(x, w, bias, out=out)


def test_nvidia_linear_binds_cublas_to_runtime_stream():
    source = (Path(__file__).parents[2] / "src/ops/linear/nvidia/linear_nvidia.cu").read_text()

    assert "cublasSetStream(handle" in source
    assert "const auto &runtime = llaisys::core::context().runtime();" in source
    assert "static_cast<cudaStream_t>(runtime.stream())" in source
    assert 'std::cerr << "cuBLAS error at "' in source


def test_nvidia_linear_caches_handles_per_thread_and_device():
    root = Path(__file__).parents[2]
    linear_source = (root / "src/ops/linear/nvidia/linear_nvidia.cu").read_text()
    runtime_source = (root / "src/core/runtime/runtime.cpp").read_text()

    assert "thread_local std::unordered_map<int, cublasHandle_t>" in linear_source
    assert "const int device_id = runtime.deviceId();" in linear_source
    assert "cublasSetStream(handle, stream)" in linear_source
    assert runtime_source.index("_api->set_device(_device_id);") < runtime_source.index("_stream = _api->create_stream();")


def test_op_linear(
    out_shape,
    x_shape,
    w_shape,
    use_bias=True,
    dtype_name="f32",
    atol=1e-5,
    rtol=1e-5,
    device_name="cpu",
    profile=False,
):
    print(f"   out {out_shape}, x {x_shape}, w {w_shape}, bias {use_bias}, dtype <{dtype_name}>")
    x, x_ = random_tensor(x_shape, dtype_name, device_name, scale=0.1)
    w, w_ = random_tensor(w_shape, dtype_name, device_name, scale=0.01)

    bias, bias_ = None, None
    if use_bias:
        bias, bias_ = random_tensor((w_shape[0],), dtype_name, device_name)

    out, out_ = random_tensor(out_shape, dtype_name, device_name)
    torch_linear(out, x, w, bias)
    llaisys.Ops.linear(out_, x_, w_, bias_)

    assert check_equal(out_, out, atol=atol, rtol=rtol)

    if profile:
        benchmark(
            lambda: torch_linear(out, x, w, bias),
            lambda: llaisys.Ops.linear(out_, x_, w_, bias_),
            device_name,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu", choices=["cpu", "nvidia"], type=str)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    test_nvidia_linear_binds_cublas_to_runtime_stream()
    test_nvidia_linear_caches_handles_per_thread_and_device()

    testShapes = [
        ((2, 3), (2, 4), (3, 4), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
    ]
    testDtypePrec = [
        # type, atol, rtol
        ("f32", 1e-5, 1e-5),
        ("f16", 1e-3, 1e-3),
        ("bf16", 1e-2, 1e-2),
    ]
    print(f"Testing Ops.linear on {args.device}")
    for shapes in testShapes:
        for dtype_name, atol, rtol in testDtypePrec:
            test_op_linear(*shapes, dtype_name, atol, rtol, args.device, args.profile)

    print("\033[92mTest passed!\033[0m\n")
