import argparse
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[2]
KERNEL_SOURCE = ROOT / "src/ops/kv_cache_move/nvidia/kv_cache_move_nvidia.cu"
OP_SOURCE = ROOT / "src/ops/kv_cache_move/op.cpp"


def torch_kv_cache_move_reference(
    out: torch.Tensor,
    inp: torch.Tensor,
    block_ids: torch.Tensor,
    cut_idx: torch.Tensor,
    pos_ids: torch.Tensor,
) -> torch.Tensor:
    """Apply the V3 block table mapping: [count, id_0, id_1, ...]."""
    result = out.clone()
    token_num = result.shape[1]

    for batch in range(block_ids.shape[0]):
        lo = int(cut_idx[batch])
        up = int(cut_idx[batch + 1])
        for seq_id in range(lo, up):
            pos_id = int(pos_ids[seq_id])
            block_id = int(block_ids[batch, pos_id // token_num + 1])
            result[block_id, pos_id % token_num].copy_(inp[seq_id])

    return result


def test_static_guards() -> None:
    """Keep the no-GPU safety and vector-copy paths from regressing."""
    kernel_source = KERNEL_SOURCE.read_text()
    op_source = OP_SOURCE.read_text()
    test_source = Path(__file__).read_text()

    assert "copy_4d(in + i, out + i)" in kernel_source
    assert "numel % kVectorWidth" in kernel_source
    assert "tensor" + ".load(" not in test_source
    assert "MemcpyKind.D2D" in test_source
    for parameter in ("token_num", "max_block_num", "max_seq_len", "numel"):
        assert f"{parameter} > 0" in op_source


def copy_to_llaisys(value: torch.Tensor, dtype, device):
    import llaisys

    tensor = llaisys.Tensor(value.shape, dtype=dtype, device=device)
    api = llaisys.RuntimeAPI(device)
    api.memcpy_sync(
        tensor.data_ptr(),
        value.data_ptr(),
        value.numel() * value.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return tensor


def copy_from_llaisys(value, dtype: torch.dtype) -> torch.Tensor:
    import llaisys
    from test_utils import device_name, torch_device

    result = torch.empty(
        value.shape(),
        dtype=dtype,
        device=torch_device(device_name(value.device_type()), value.device_id()),
    )
    api = llaisys.RuntimeAPI(value.device_type())
    api.memcpy_sync(
        result.data_ptr(),
        value.data_ptr(),
        result.numel() * result.element_size(),
        llaisys.MemcpyKind.D2D,
    )
    return result


def test_kv_cache_move(dtype_name: str) -> None:
    import llaisys
    from test_utils import llaisys_dtype

    torch_dtype = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
    }[dtype_name]
    device = torch.device("cuda:0")
    llaisys_device = llaisys.DeviceType.NVIDIA

    # numel=8 keeps every row 4-element aligned for all tested dtypes.
    token_num, nkvh, dh = 4, 2, 4
    inp = torch.arange(5 * nkvh * dh, dtype=torch.float32, device=device).reshape(5, nkvh, dh).to(torch_dtype)
    out = torch.zeros((7, token_num, nkvh, dh), dtype=torch_dtype, device=device)

    # Two V3 rows: count prefix plus non-contiguous physical block IDs.
    block_ids = torch.tensor([[2, 5, 1], [4, 6, 3]], dtype=torch.int64, device=device)
    cut_idx = torch.tensor([0, 2, 5], dtype=torch.int64, device=device)
    # Batch 0 writes after a prefix; batch 1 spans both logical blocks.
    pos_ids = torch.tensor([5, 6, 1, 2, 7], dtype=torch.int64, device=device)
    vector_bytes = 4 * inp.element_size()
    # seq_id=0 maps to physical block 1, token 1. This pair takes copy_4d.
    assert inp[0].data_ptr() % vector_bytes == 0
    assert out[1, 1].data_ptr() % vector_bytes == 0
    expected = torch_kv_cache_move_reference(out, inp, block_ids, cut_idx, pos_ids)

    out_ll = copy_to_llaisys(out, llaisys_dtype(dtype_name), llaisys_device)
    inp_ll = copy_to_llaisys(inp, llaisys_dtype(dtype_name), llaisys_device)
    block_ids_ll = copy_to_llaisys(block_ids, llaisys.DataType.I64, llaisys_device)
    cut_idx_ll = copy_to_llaisys(cut_idx, llaisys.DataType.I64, llaisys_device)
    pos_ids_ll = copy_to_llaisys(pos_ids, llaisys.DataType.I64, llaisys_device)

    llaisys.Ops.kv_cache_move(out_ll, inp_ll, block_ids_ll, cut_idx_ll, pos_ids_ll, max_seq_len=3)

    actual = copy_from_llaisys(out_ll, torch_dtype)
    assert torch.equal(actual, expected), f"kv_cache_move mismatch for {dtype_name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--static", action="store_true", help="Run source-level checks without CUDA.")
    parser.add_argument("--device", default="nvidia", choices=["nvidia"])
    args = parser.parse_args()

    test_static_guards()
    if args.static:
        print("Static kv_cache_move checks passed.")
        sys.exit(0)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required; use --static for a no-GPU check.")

    for dtype_name in ("f32", "f16", "bf16"):
        test_kv_cache_move(dtype_name)
    print("\033[92mkv_cache_move tests passed!\033[0m")
