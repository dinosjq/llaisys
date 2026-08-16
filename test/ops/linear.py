import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import torch
import llaisys
from test_utils import random_tensor, check_equal, benchmark


def torch_linear(out, x, w, bias):
    torch.nn.functional.linear(x, w, bias, out=out)


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

    testShapes = [
        ((2, 3), (2, 4), (3, 4), True),
        ((512, 4096), (512, 4096), (4096, 4096), True),
    ]
    # Small-M shapes exercise the NVIDIA hybrid GEMV path (1 <= M <= 15).
    if args.device == "nvidia":
        testShapes.extend(
            [
                ((1, 1536), (1, 1536), (1536, 1536), True),
                ((4, 8960), (4, 1536), (8960, 1536), False),
                ((8, 1536), (8, 8960), (1536, 8960), True),
            ]
        )

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

    # Force the GEMV backend where the shape is supported.
    if args.device == "nvidia":
        os.environ["LLAISYS_LINEAR_BACKEND"] = "gemv"
        print("Testing Ops.linear GEMV backend (LLAISYS_LINEAR_BACKEND=gemv)")
        for shapes in testShapes[2:]:  # small-M shapes only
            for dtype_name, atol, rtol in testDtypePrec:
                test_op_linear(*shapes, dtype_name, atol, rtol, args.device, False)
        os.environ.pop("LLAISYS_LINEAR_BACKEND", None)

    print("\033[92mTest passed!\033[0m\n")
