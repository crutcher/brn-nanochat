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

# training example

- the iterator stream for samples may be quite large.
- training a `nanochat` equivalent tokenizer takes ~80 CPU minutes.

```rust,no_run
use wordchuck::training::trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
use wordchuck::vocab::io::tiktoken_io::save_word_map_to_tiktoken_path;
use wordchuck::vocab::public::patterns::GPT4_PATTERN;
use wordchuck::vocab::UnifiedTokenVocab;
use wordchuck::encoders::UnifiedVocabEncoder;
use wordchuck::decoders::DictionaryDecoder;
use wordchuck::rayon::{ParallelRayonEncoder, ParallelRayonDecoder};
use std::sync::Arc;

fn example<I, S>(
    vocab_size: usize,
    batches: I,
    tiktoken_save_path: Option<String>,
) where
    I: IntoIterator,
    I::Item: AsRef<[S]>,
    S: AsRef<str>,
{
    // We can pick any unsigned integer type > vocab_size;
    // See [`wordchuck::types::TokenType`].
    type T = u32;
    type K = String;
    type C = u64;

    let options = BinaryPairVocabTrainerOptions::new(
        GPT4_PATTERN,
        vocab_size,
    );

    let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();

    for batch in batches {
        // The trainer has no parallelism.
        // The perceived benefits of parallelism in the trainer
        // are insignificant if the IO for the sample source is
        // fed by another thread.
        trainer.update_from_samples(batch.as_ref());
    }

    let vocab: Arc<UnifiedTokenVocab<T>> = trainer
        .train::<T>()
        .expect("training failed")
        .into();

    if let Some(path) = tiktoken_save_path {
        save_word_map_to_tiktoken_path(&vocab.word_vocab, &path)
            .expect("failed to save tiktoken vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: UnifiedVocabEncoder<T> = UnifiedVocabEncoder::<T>::new(vocab.clone());
    let encoder = ParallelRayonEncoder::new(encoder);

    let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
    let decoder = ParallelRayonDecoder::new(decoder);
}
```

# Example Tokenizer Trainer

Each shard is ~90MB parquet file.

- 128/64 Core Thread Ripper

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset  --time-encode-decode 
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/wordchuck/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 1.34s
     Running `target/release/tokenizer_trainer --dataset-dir /media/Data/nanochat/dataset --time-encode-decode`
Loading Shards: [0, 1, 2, 3, 4, 5, 6, 7]
...

Training Tokenizer on shards: [0, 1, 2, 3, 4, 5, 6, 7]
- shard: 0
- shard: 1
- shard: 2
- shard: 3
- shard: 4
- shard: 5
- shard: 6
- shard: 7
- train
- training_duration: 220.05s
- vocab_size: 65535

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512

Timing Encode:
- batch avg: 69.918721ms
- sample avg: 136.56µs
- avg bps: 34.72 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 24645141
- sample byte/token: 3.94

Timing Decode:
- batch avg: 2.373206ms
- sample avg: 4.635µs

real    3m45.018s
user    78m36.407s
sys     37m53.941s
```
