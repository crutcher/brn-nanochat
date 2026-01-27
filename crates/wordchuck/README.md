# rust-centric clone of nanochat/rustbpe

This is a high-performance rust BPE tokenizer trainer/encoder/decoder.

It is inspired by [nanochat rustbpe](https://github.com/karpathy/nanochat/tree/master/rustbpe)

The current status is productionization towards an alpha release.

TODO:

- New Name / New Repo. ( `wordchuck` conflicts, alas)
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
use wordchuck::training::bpe_trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
use wordchuck::vocab::io::tiktoken_io::save_span_map_to_tiktoken_path;
use wordchuck::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
use wordchuck::vocab::{ByteVocab, UnifiedTokenVocab};
use wordchuck::encoders::MergeHeapVocabEncoder;
use wordchuck::decoders::DictionaryDecoder;
use wordchuck::rayon::{ParallelRayonEncoder, ParallelRayonDecoder};
use wordchuck::regex::default_regex_supplier;
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
        OA_GPT3_CL100K_WORD_PATTERN,
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

    let byte_vocab: Arc<ByteVocab<T>> = Arc::new(Default::default());

    let vocab: Arc<UnifiedTokenVocab<T>> = trainer
        .train(byte_vocab.clone())
        .expect("training failed")
        .into();

    if let Some(path) = tiktoken_save_path {
        save_span_map_to_tiktoken_path(&vocab.span_vocab.span_map(), &path)
            .expect("failed to save tiktoken vocab");
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: MergeHeapVocabEncoder<T> = MergeHeapVocabEncoder::<T>::init(
        vocab.clone(),
        default_regex_supplier
    );
    let encoder = ParallelRayonEncoder::new(encoder);

    let decoder = DictionaryDecoder::from_unified_vocab(vocab.clone());
    let decoder = ParallelRayonDecoder::new(decoder);
}
```

# Example Tokenizer Trainer

Each shard is ~90MB parquet file.

- 64 Core Thread Ripper

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset --time-encode-decode 
   Compiling wordchuck v0.0.6 (/home/crutcher/git/brn-nanochat/crates/wordchuck)
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/wordchuck/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 1.85s
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
- training_duration: 203.40s
- vocab_size: 65535

Samples Summary:
- count: 20480
- avg size: 4741

Timing Config:
- batch size: 512

Timing Encode:
- batch avg: 76.543966ms
- sample avg: 149.499µs
- avg bps: 31.71 MB/s

Observed Bytes/Token Stats:
- total bytes: 97103222
- total tokens: 24645141
- sample byte/token: 3.94

Timing Decode:
- batch avg: 2.466106ms
- sample avg: 4.816µs

real    3m28.924s
user    6m37.652s
sys     0m35.035s
```
