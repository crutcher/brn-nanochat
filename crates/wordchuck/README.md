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

```rust,ignore
    let vocab: Arc<UnifiedTokenVocab<T>> =
        BinaryPairVocabTrainer::new_with_vocab_size(args.vocab_size)
            .train_vocab_from_sample_iter::<T, K, C, _>(samples)
            .expect("training failed")
            .extend_word_vocab_from_pair_vocab()
            .into();

    let training_duration = std::time::Instant::now().duration_since(t0);
    println!("- training_duration: {:#?}", training_duration);
    println!("- vocab_size: {:#?}", vocab.max_token());
    
    if let Some(path) = args.tiktoken_save_path {
        vocab.word_vocab.save_to_tiktoken_path(&path)?;
        println!("- tiktoken vocab: {path:?}");
    }

    let encoder: UnifiedVocabEncoder<T> = UnifiedVocabEncoder::<T>::new(vocab.clone());
    let encoder = ParallelEncoder::new(encoder);

    let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
    let decoder = ParallelDecoder::new(decoder);
```

# training and timing

- Note: my machine is a beast (64-core Threadripper; NVME data disk).

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset --shards ..8 --vocab-size=65536 --time-encode-decode --batch-size 512 --num-timing-batches 60
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/bpetok/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 1.54s
     Running `target/release/tokenizer_trainer --dataset-dir /media/Data/nanochat/dataset --shards ..8 --vocab-size=65536 --time-encode-decode --batch-size 512 --num-timing-batches 60`
Loading Shards ...: [0, 1, 2, 3, 4, 5, 6, 7]

Training Tokenizer on shards: [0, 1, 2, 3, 4, 5, 6, 7]
- training_duration: 74.15810139s
- vocab_size: 65535
- size_estimate: 917613

Samples Summary:
- count: 53248
- avg size: 4783

Timing Config:
- batch size: 512

Timing CPSEncoder Encode:
- batch avg: 83.835533ms
- sample avg: 163.741µs
- avg bps: 29.21 MB/s

Timing Decode: ExpansionDecoder
- decoder est bytes: 1566720
- batch avg: 2.219528ms
- sample avg: 4.335µs

Timing Decode: DictionaryDecoder
- decoder est bytes: 1860233
- batch avg: 1.463183ms
- sample avg: 2.857µs

Timing Decode: CorpusDecoder
- decoder est bytes: 1820714
- batch avg: 1.485641ms
- sample avg: 2.901µs

real    1m26.091s
user    86m4.472s
sys     27m10.539s
```
