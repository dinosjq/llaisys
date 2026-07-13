  period: prefill   step: 0   total_ms: 274.49 ms

    operator            avg_ms    count    total_ms     %
    ──────────────────────────────────────────────────────
    linear:k_proj         4.97       28      139.24   50.7
    attn_norm             2.90       28       81.22   29.6
    linear:v_proj         0.36       28       10.00    3.6
    top_k                 6.26        1        6.26    2.3
    swiglu                0.20       28        5.61    2.0
    linear:up_proj        0.20       28        5.48    2.0
    add:mlp               0.18       28        5.14    1.9
    add:attn              0.09       28        2.45    0.9
    rope:q                0.08       28        2.18    0.8
    linear:q_proj         0.07       28        2.03    0.7
    linear:o_proj         0.07       28        2.01    0.7
    kv_cache_move:k       0.07       28        1.89    0.7
    kv_cache_move:v       0.07       28        1.84    0.7
    linear:gate_proj      0.07       28        1.89    0.7
    linear:down_proj      0.07       28        1.87    0.7
    mlp_norm              0.07       28        1.86    0.7
    rope:k                0.07       28        1.98    0.7
    paged_attn            0.07       28        1.84    0.7
    embed                 0.16        1        0.16    0.1
    embed:gather          0.05        1        0.05    0.0
    out_norm              0.06        1        0.06    0.0
    linear:lm_head        0.05        1        0.05    0.0
    ──────────────────────────────────────────────────────
    total                                           100.0

  period: decode   step: 64   total_ms: 50.54 ms

    operator            avg_ms    count    total_ms     %
    ──────────────────────────────────────────────────────
    linear:o_proj         0.17       28        4.63    9.2
    add:mlp               0.18       28        5.15   10.2
    swiglu                0.18       28        5.10   10.1
    linear:up_proj        0.18       28        5.12   10.1
    top_k                 3.29        1        3.29    6.5
    linear:k_proj         0.09       28        2.52    5.0
    add:attn              0.09       28        2.55    5.0
    linear:v_proj         0.07       28        2.04    4.0
    rope:q                0.07       28        2.05    4.1
    rope:k                0.07       28        1.96    3.9
    attn_norm             0.07       28        1.93    3.8
    linear:q_proj         0.07       28        1.94    3.8
    linear:gate_proj      0.07       28        1.93    3.8
    linear:down_proj      0.07       28        1.94    3.8
    kv_cache_move:k       0.07       28        1.91    3.8
    kv_cache_move:v       0.07       28        1.96    3.9
    paged_attn            0.07       28        1.96    3.9
    mlp_norm              0.07       28        1.87    3.7
    embed                 0.51        1        0.51    1.0
    embed:gather          0.06        1        0.06    0.1
    out_norm              0.06        1        0.06    0.1
    linear:lm_head        0.06        1        0.06    0.1
    ──────────────────────────────────────────────────────
    total                                           100.0
