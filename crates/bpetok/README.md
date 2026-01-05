# rust-centric clone of nanochat/rustbpe

See: [nanochat rustbpe](https://github.com/karpathy/nanochat/tree/master/rustbpe)

This repo aims to be a rust-first *BPETokenizer* library;
focusing on performance and ease of use as a first-class rust crate.

Python bindings already exist for `nanochat/rustbpe`.

## Status: WIP

I am incrementally porting features from `nanochat/rustbpe` to this crate;
while cleaning up the rust mechanics, and writing full tests and docs.

This is complete to training, tokenization, and decoding.

TODO:

- Save/Load vocabularies.
    - Save/Load well-known / named remote vocabularies.
    - Save/Load to `tiktoken` vocab format.
- Benchmarks.
- Error handling (as `Result`s, not panics).
- Tuning
    - Instrument `tiktoken` (via `tracing`).
    - Compare / fix perf differences.
- Python/C*/Java Bindings?

See:

- [examples/tokenizer_trainer](examples/tokenizer_trainer)
