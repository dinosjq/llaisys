#!/bin/bash
cd /home/songjq/llaisys-main
NCU=/usr/local/cuda/bin/ncu
for tag in v4_h6t8c1 v4_h6t8c4 v5_h6t8c1 v5_h3t16c4; do
  echo "=== $tag ==="
  $NCU --import /tmp/ncu_${tag}.ncu-rep --page summary 2>&1 | head -80
  echo ""
done
