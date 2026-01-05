# Example Tokenizer Trainer

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

- we don't support export / load of the tokenizer yet; so this is all in-memory for nothing.
- the nanochat choice of 2**16 vocab size *seems* cool; but it ignores:
    - nighter the `rustbpe` nor `tiktoken` tokenizers support tuning the `Token` width for a smaller vocab size.
    - the token=>embedding table also doesn't tune for smaller vocab sizes.

See: [original nanochat/speedrun.sh](https://github.com/karpathy/nanochat/blob/master/speedrun.sh#L58C1-L69C51)

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

# training and timing

- Note: my machine is a beast (64-core Threadripper; NVME data disk).

```terminaloutput
$ time cargo run --release -p tokenizer_trainer -- --dataset-dir /media/Data/nanochat/dataset --shards ..8 --vocab-size=65536 --time-encode-decode --batch-size 256 --tiktoken-save-path /home/crutcher/junk/nanochat.tiktoken
   Compiling bpetok v0.0.1 (/home/crutcher/git/brn-nanochat/crates/bpetok)
   Compiling tokenizer_trainer v0.0.0 (/home/crutcher/git/brn-nanochat/crates/bpetok/examples/tokenizer_trainer)
    Finished `release` profile [optimized] target(s) in 1.99s
     Running `target/release/tokenizer_trainer --dataset-dir /media/Data/nanochat/dataset --shards ..8 --vocab-size=65536 --time-encode-decode --batch-size 256 --tiktoken-save-path /home/crutcher/junk/nanochat.tiktoken`
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
    vocab_size: 65536,
    time_encode_decode: true,
    batch_size: 256,
    tiktoken_save_path: Some(
        "/home/crutcher/junk/nanochat.tiktoken",
    ),
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
Loading Shards ...: [0, 1, 2, 3, 4, 5, 6, 7]

Training Tokenizer on shards: [0, 1, 2, 3, 4, 5, 6, 7]
- training_duration: 76.070333518s
- vocab_size: 65535
- size_estimate: 3822326
- tiktoken vocab: "/home/crutcher/junk/nanochat.tiktoken"

Samples Summary:
- count: 8192
- avg size: 4712

Timing Config:
- batch size: 256

Timing CPSEncoder Encode:
- batch avg: 182.233µs
- sample avg: 22ns

Timing Decode: ExpansionDecoder
- decoder est bytes: 1566720
- batch avg: 2.10314ms
- sample avg: 8.215µs

Timing Decode: DictionaryDecoder
- decoder est bytes: 1860233
- batch avg: 1.343481ms
- sample avg: 5.247µs

Timing Decode: CorpusDecoder
- decoder est bytes: 1820714
- batch avg: 1.248797ms
- sample avg: 4.878µs

real	1m20.341s
user	81m59.549s
sys	25m15.449s

$ head -5 /home/crutcher/junk/nanochat.tiktoken 
IHQ= 256
dGg= 257
IGE= 258
aGU= 259
aW4= 260

$ tail -5 /home/crutcher/junk/nanochat.tiktoken 
IGdlb3A= 65531
YWxsaW9ucw== 65532
YWxsaW0= 65533
b25vbWlzdA== 65534
ZGlhbWV0ZXI= 65535
```
