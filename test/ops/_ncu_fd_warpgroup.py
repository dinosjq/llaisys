"""ncu 采集「warp 分组终版」flash_decoding_parallel_kernel 的驱动负载。

用法（在 venv 下）：
    python test/ops/_ncu_fd_warpgroup.py --mode big    # batch=64 × seq2048（dispatch: WN2/TM2/CO8）
    python test/ops/_ncu_fd_warpgroup.py --mode mid    # batch=16 × seq2048（dispatch: WN3/TM3/CO2）
    python test/ops/_ncu_fd_warpgroup.py --mode small  # batch=1  × seq512 （dispatch: WN3/TM3/CO1）

模型口径：Qwen2-1.5B GQA（nh=12/nkvh=2 → rep=6, hd=128, bf16），KV_CACHE_BLOCK_SIZE=64。
走真实 op 路径 llaisys.Ops.paged_attention(..., is_prefill=False) → flash_decoding dispatch。
仅做测量负载，不做数值正确性（正确性由 test/ops/flash_decoding.py 覆盖）。
"""
import sys
import os
import argparse

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)
import llaisys
import torch
from ctypes import c_void_p
from test_utils import torch_device, llaisys_device, llaisys_dtype

TOKEN_NUM = 64  # KV_CACHE_BLOCK_SIZE
NH, NKVH, HD = 12, 2, 128
DTYPE_NAME = "bf16"


def sm89_dispatch_config(tot_block_num, batch):
    """复刻 flash_decoding_nvidia.cu 中 sm_89（非 Blackwell）选参规则，仅用于打印报告。"""
    tot = tot_block_num
    wn = tm = 3 if tot <= 512 else 2
    tk = 16
    co = 1 if tot <= 128 else (2 if tot <= 512 else (4 if tot <= 1024 else 8))
    return wn, tm, tk, co


def build_case(batch, seq_per_sample):
    assert seq_per_sample % TOKEN_NUM == 0
    block_num = seq_per_sample // TOKEN_NUM
    device_id = 0
    dev = torch_device("nvidia", device_id)
    ld = llaisys_device("nvidia")

    q = torch.randn((batch, NH, HD), dtype=torch.bfloat16, device=dev)
    q_ll = llaisys.Tensor((batch, NH, HD), dtype=llaisys.DataType.BF16, device=ld, device_id=device_id)
    q_ll.load(c_void_p(q.data_ptr()))

    # 每 sample 独占连续 block（避免 L2 复用污染），block_ids V3 前缀和格式
    total_blocks = batch * block_num
    k_cache = torch.randn((total_blocks, TOKEN_NUM, NKVH, HD), dtype=torch.bfloat16, device=dev)
    v_cache = torch.randn((total_blocks, TOKEN_NUM, NKVH, HD), dtype=torch.bfloat16, device=dev)
    k_ll = llaisys.Tensor((total_blocks, TOKEN_NUM, NKVH, HD), dtype=llaisys.DataType.BF16, device=ld, device_id=device_id)
    v_ll = llaisys.Tensor((total_blocks, TOKEN_NUM, NKVH, HD), dtype=llaisys.DataType.BF16, device=ld, device_id=device_id)
    k_ll.load(c_void_p(k_cache.data_ptr()))
    v_ll.load(c_void_p(v_cache.data_ptr()))

    mb = block_num + 1  # +1 前缀列
    rows, cum = [], 0
    for i in range(batch):
        cum += block_num
        bids = [i * block_num + j for j in range(block_num)]
        rows.append([cum] + bids + [0] * (mb - len(bids) - 1))
    bids_t = torch.tensor(rows, dtype=torch.int64, device=dev)
    bids_ll = llaisys.Tensor((batch, mb), dtype=llaisys.DataType.I64, device=ld, device_id=device_id)
    bids_ll.load(c_void_p(bids_t.data_ptr()))

    cut_idx = list(range(batch + 1))  # 每 sample seqlen=1
    cut_t = torch.tensor(cut_idx, dtype=torch.int64, device=dev)
    cut_ll = llaisys.Tensor((batch + 1,), dtype=llaisys.DataType.I64, device=ld, device_id=device_id)
    cut_ll.load(c_void_p(cut_t.data_ptr()))

    tot_t = torch.tensor([seq_per_sample] * batch, dtype=torch.int64, device=dev)
    tot_ll = llaisys.Tensor((batch,), dtype=llaisys.DataType.I64, device=ld, device_id=device_id)
    tot_ll.load(c_void_p(tot_t.data_ptr()))

    attn_val = llaisys.Tensor((batch, NH, HD), dtype=llaisys.DataType.BF16, device=ld, device_id=device_id)
    attn_acc = llaisys.Tensor((total_blocks, NH, HD), dtype=llaisys.DataType.F32, device=ld, device_id=device_id)
    attn_sum = llaisys.Tensor((total_blocks, NH, 1), dtype=llaisys.DataType.F32, device=ld, device_id=device_id)
    attn_max = llaisys.Tensor((total_blocks, NH, 1), dtype=llaisys.DataType.F32, device=ld, device_id=device_id)
    scale = 1.0 / (HD ** 0.5)

    return {
        "batch": batch, "seq": seq_per_sample, "block_num": block_num,
        "total_blocks": total_blocks, "attn_val": attn_val, "q": q_ll,
        "k": k_ll, "v": v_ll, "bids": bids_ll, "cut": cut_ll, "tot": tot_ll,
        "scale": scale, "acc": attn_acc, "sum": attn_sum, "max": attn_max,
    }


def run_once(c):
    llaisys.Ops.paged_attention(
        c["attn_val"], c["q"], c["k"], c["v"], c["bids"], c["cut"], c["tot"],
        c["total_blocks"], c["scale"], False, c["acc"], c["sum"], c["max"],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="big", choices=["big", "mid", "small"])
    ap.add_argument("--iters", type=int, default=40)
    args = ap.parse_args()

    loads = {
        "big": (64, 2048),
        "mid": (16, 2048),
        "small": (1, 512),
    }
    batch, seq = loads[args.mode]
    c = build_case(batch, seq)
    wn, tm, tk, co = sm89_dispatch_config(c["total_blocks"], batch)
    group_num = (6 + tm - 1) // tm  # HEAD_NUM=rep=6
    block_threads = group_num * wn * 32
    grid = (c["total_blocks"] // co + batch, NKVH)
    print(f"[load] mode={args.mode} batch={batch} seq={seq} block_num={c['block_num']} "
          f"bs={TOKEN_NUM} total_blocks={c['total_blocks']} dtype={DTYPE_NAME} nh={NH} nkvh={NKVH} hd={HD}")
    print(f"[dispatch sm_89] WN={wn} TM={tm} TK={tk} CO={co} blockDim={block_threads}thr "
          f"grid={grid} head_num(rep)=6")
    sys.stdout.flush()

    # 首轮校验：kernel 正常 launch 且无异常（数值正确性由 test/ops/flash_decoding.py 覆盖）
    run_once(c)
    torch.cuda.synchronize()
    print(f"[sanity] first launch OK (no error)")
    sys.stdout.flush()

    for i in range(args.iters):
        run_once(c)
    torch.cuda.synchronize()

    # 普通时钟下的整 op（parallel + reduce）cudaEvent 参考耗时，供与 ncu base 时钟对照
    ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    ev0.record()
    for _ in range(args.iters):
        run_once(c)
    ev1.record()
    torch.cuda.synchronize()
    us = ev0.elapsed_time(ev1) * 1000.0 / args.iters
    print(f"[cudaEvent ref] whole op avg = {us:.1f} us/iter (iters={args.iters}, normal clocks)")
    print(f"[done] ran {args.iters} iterations")


if __name__ == "__main__":
    main()
