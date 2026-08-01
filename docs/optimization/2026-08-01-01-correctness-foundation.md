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

These changes are correctness-oriented and should be performance-neutral apart from bounded CPU sampling work.
There is no public request seed API; deterministic RNG injection is internal-test-only. Roll back this task commit
to restore the previous request and sampling behavior.
