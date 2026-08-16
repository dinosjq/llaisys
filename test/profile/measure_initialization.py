#!/usr/bin/env python3
"""Measure LLAISYS model initialization time and GPU memory usage."""

import argparse
import json
import time
from pathlib import Path

import llaisys
import torch
from llaisys import libllaisys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model directory")
    parser.add_argument("--output", required=True, help="Path to output JSON")
    args = parser.parse_args()

    torch.cuda.synchronize()
    free_before, total = torch.cuda.mem_get_info()

    start = time.perf_counter()
    model = llaisys.models.load_causal_lm(args.model, llaisys.DeviceType.NVIDIA)
    try:
        torch.cuda.synchronize()
        initialization_wall_seconds = time.perf_counter() - start
        free_after, _ = torch.cuda.mem_get_info()

        result = {
            "llaisys_module": llaisys.__file__,
            "shared_library": libllaisys.LIB_LLAISYS._name,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_total_mib": total / 2**20,
            "initialization_wall_seconds": initialization_wall_seconds,
            "initialization_gpu_memory_delta_mib": (free_before - free_after) / 2**20,
            "free_before_mib": free_before / 2**20,
            "free_after_mib": free_after / 2**20,
        }

        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
    finally:
        model.close()


if __name__ == "__main__":
    main()
