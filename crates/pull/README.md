# Shard Pull Utility

Each shard is ~90MB parquet file.

```terminaloutput
$ cargo run --release -p pull -- --dataset-dir /media/data/nanochat/dataset --shards ..8
```

# nanochat-equivalent tokenizer train

_TL;DR_:

    I think 2**16 vocab size is a bad choice for nanochat.
    It was picked arbitrarily, without tracing out the full tuning.

    It should either be smaller (to reserve space for meta tokens),
    and actually use that size to reduce memory costs / vector sizes.

    Or, we should search over larger sizes.

This is a *partial example* of training a nanochat-equivalent tokenizer:

```terminaloutput
# Build the `pull` binary; and pre-download the first 8 shards of the dataset.
$ cargo run --release -p pull -- --dataset-dir /media/Data/nanochat/dataset --shards ..8
# Train the tokenizer on the first 8 shards of the dataset.
$ time cargo run --release -p pull -- --dataset-dir /media/Data/nanochat/dataset --shards ..8 --train-tokenizer --vocab-size=65536
    Finished `release` profile [optimized] target(s) in 0.32s
     Running `target/release/pull --dataset-dir /media/Data/nanochat/dataset --shards ..8 --train-tokenizer --vocab-size=65536`
Args {
    shards: [
        Slice {
            start: 0,
            end: Some(
                8,
            ),
            step: 1,
        },
    ],
    dataset_dir: "/media/Data/nanochat/dataset",
    train_tokenizer: true,
    vocab_size: 65536,
}
DatasetCacheConfig {
    cache_dir: "/media/Data/nanochat/dataset",
    source: DatasetSource {
        base_url: "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard: 1822,
        index_pad_width: 5,
        shard_template: "shard_{index}.parquet",
    },
}
vocab_size: 65536

real    1m14.608s
user    75m21.630s
sys     24m21.012s
```

- Note: my machine is a beast (64 core Threadripper; NVME data disk).
- we don't support export / load of the tokenizer yet; so this is all in-memory for nothing.
- the nanochat choice of 2**16 vocab size *seems* cool; but it ignores:
    - nighter the `rustbpe` nor `tiktoken` tokenizers support tuning the `Token` width for a smaller vocab size.
    - the token=>embedding table also doesn't tune for smaller vocab sizes.

See: [nanochat/speedrun.sh](https://github.com/karpathy/nanochat/blob/master/speedrun.sh#L58C1-L69C51)

```bash
# Download the first ~2B characters of pretraining dataset
# look at dev/repackage_data_reference.py for details on how this data was prepared
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# See comment below for why 240 is the right number here
python -m nanochat.dataset -n 240 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**16 = 65536 on ~2B characters of data
python -m scripts.tok_train --max_chars=2000000000
```

# benchmarks

Same as above:

```terminaloutput
# Build the `pull` binary; and pre-download the first 8 shards of the dataset.
$ cargo run --release -p pull -- --dataset-dir /media/Data/nanochat/dataset --shards ..8
$ time cargo run --release -p pull -- --dataset-dir /media/Data/nanochat/dataset --shards ..8 --train-tokenizer --vocab-size=65536 --time-encode-decode
   Compiling bpetok v0.0.1 (/home/crutcher/git/brn-nanochat/crates/bpetok)
   Compiling pull v0.0.0 (/home/crutcher/git/brn-nanochat/crates/pull)
    Finished `release` profile [optimized] target(s) in 1.61s
     Running `target/release/pull --dataset-dir /media/Data/nanochat/dataset --shards ..8 --train-tokenizer --vocab-size=65536 --time-encode-decode`
Args {
    shards: [
        Slice {
            start: 0,
            end: Some(
                8,
            ),
            step: 1,
        },
    ],
    dataset_dir: "/media/Data/nanochat/dataset",
    train_tokenizer: true,
    vocab_size: 65536,
    time_encode_decode: true,
}
DatasetCacheConfig {
    cache_dir: "/media/Data/nanochat/dataset",
    source: DatasetSource {
        base_url: "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main",
        max_shard: 1822,
        index_pad_width: 5,
        shard_template: "shard_{index}.parquet",
    },
}

Training Tokenizer on shards: [0, 1, 2, 3, 4, 5, 6, 7]
- training_duration: 72.612455275s
- vocab_size: 65536
- size_estimate: 917613

Training DictionaryDecoder:
- training_duration: 9.944697ms
- size_estimate: (
    1572864,
    293769,
)

Training CorpusDecoder:
- training_duration: 24.811911ms
- size_estimate: (
    1566720,
    253994,
)

Timing Encode:
- sample count: 1024
- avg sample size: 4108
- encode avg duration: 4.391576ms

DictionaryDecoder decode_to_string avg duration: 28.463µs

CorpusDecoder decode_to_string avg duration: 23.298µs

real    1m19.331s
user    80m31.884s
sys     25m26.822s
```
