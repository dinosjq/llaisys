# Correct batched inference semantics

## Confirmed defects

- Prefix-cache collision verification compared every cached token with candidate index one.
- Sampling treated nucleus candidates uniformly and did not preserve the first token crossing `top_p`.
- Chunked prefill stopped before scheduling a queue-front prompt larger than the remaining token budget.
- The legacy prompt-hash request API shared one `Sequence` between identical prompts.

## Design and API

`sampling.hpp` centralizes bounded top-k, temperature softmax, nucleus filtering, and weighted sampling. Each
`Sequence` owns an RNG, and a batch runs device top-k once at the maximum requested K before CPU sampling applies
the row-specific K.

`LlaisysQwen2Request` is an opaque handle with submit, await, abort, and release entry points. It owns an
independent `Sequence`; cached KV blocks can still be shared only through `BlockManager`.

## Tests

Core regressions cover elementwise prefix matching, complete-block reuse, deterministic greedy/temperature/top-p/
weighted sampling, and multi-round chunked prefill progress. Server integration additionally exercises concurrent
identical prompts and cancellation isolation when a model is available.

## Behavior and compatibility

Python generation and server streaming use request handles and release them for normal completion, exceptions, and
cancellation. The exported `llaisysQwen2ModelInfer` and `llaisysQwen2ModelAbort` APIs remain as deprecated
prompt-keyed compatibility paths and retain their identical-prompt limitation.

## Performance, limitations, rollback

The NVIDIA top-k kernel now uses deterministic serial selection to correct the FP16 ordering defect discovered by
the regression suite. This is intentionally a correctness-first trade-off and may reduce top-k throughput for large
K. There is no public request seed API; deterministic RNG injection is internal-test-only. Roll back the Task 2
commits to restore the previous request and sampling behavior.

## Review fixes

Cancellation now shields the active blocking `RequestAwait` task and defers releasing its opaque C handle until
that task finishes after abort. Explicit `temperature=0` is preserved by both server endpoints. Sequence
construction validates a non-null, non-empty prompt before dereferencing token storage. The regression suite also
covers mixed per-request top-k and cancellation isolation for concurrent identical prompts.

## Second review fixes

The serialized CUDA fallback was removed. The NVIDIA implementation is restored exactly to the reviewed
`960088c` parallel radix kernel; three fresh rebuild-and-install runs of the full NVIDIA top-k suite passed for
FP32, FP16, BF16, batched rows, and K through the existing generic coverage. Request submission now performs the
model submit before allocating its C wrapper and releases the submitted request if wrapper construction fails.

The rebuilt isolated-library BF16 smoke for `(1, 151936), K=100` used 10 warmups, 20 CUDA-event measurements,
and explicit synchronization: values matched `torch.topk`; median latency was `1.2815 ms` and minimum latency was
`1.2780 ms`.
