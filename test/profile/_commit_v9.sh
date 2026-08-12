#!/bin/bash
cd /home/songjq/llaisys-main
git add src/ops/paged_attention/nvidia/flash_decoding_v9_nvidia.cu test/profile/_test_v9.py test/profile/_bench_v9.py test/profile/_ncu_v9.py
git add -f docs/optimization/2026-08-12-flash-decoding-v9-merge.md
git commit -m "feat: v9 merge experiment"
