use arrow::array::StringArray;
use bpetok::{TokenDecoder, Tokenizer, TokenizerOptions};
use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use compact_str::CompactString;
use nanochat_data::dataset::DatasetCacheConfig;
use std::collections::HashSet;
use std::time::Duration;

/// Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// Train a tokenizer.
    #[arg(long, default_value = "false")]
    pub train_tokenizer: bool,

    /// Vocab size.
    #[arg(long, default_value = "1000000")]
    pub vocab_size: usize,

    /// Time the avg encode/decode.
    #[arg(long, default_value = "false")]
    pub time_encode_decode: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    println!("{:#?}", args);

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir);
    println!("{:#?}", cache_config);

    let shards: Vec<usize> = {
        let max_shard = cache_config.source.max_shard;
        let mut collected: HashSet<usize> = HashSet::new();
        for slice in &args.shards {
            for idx in slice.into_iter() {
                let shard = idx.expect_elem_index(max_shard);
                collected.insert(shard);
            }
        }
        let mut shards: Vec<usize> = collected.into_iter().collect();
        shards.sort();
        shards
    };

    let mut cache = cache_config.init()?;

    cache.load_shards(&shards)?;

    if args.train_tokenizer {
        type T = u32;
        type C = u32;
        type K = CompactString;

        let options = TokenizerOptions::with_capacity(args.vocab_size);

        println!("training tokenizer: {:#?}", shards);
        let t0 = std::time::Instant::now();

        let cache_ref = &cache;

        // Note: rather than repeating this, this should be a func to generate the Iterator.
        // But I can't work out the lifetimes.
        let samples = shards.clone().into_iter().flat_map(move |shard| {
            cache_ref
                .read_cached_batches(shard)
                .expect("failed to read batch")
                .flat_map(|batch| {
                    batch
                        .iter()
                        .flat_map(|sample| {
                            let text = sample
                                .column(0)
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap();
                            text.iter()
                                .map(|s| s.unwrap().to_string())
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>()
                })
        });

        let tokenizer: Tokenizer<T> = options.train_from_sample_iterator::<T, K, C, _>(samples);
        let t1 = std::time::Instant::now();
        let training_duration = t1.duration_since(t0);
        println!("tokenizer training_duration: {:#?}", training_duration);
        println!("tokenizer.vocab_size: {:#?}", tokenizer.vocab_size());
        println!("tokenizer.size_estimate: {:#?}", tokenizer.size_estimate());

        println!("training DictionaryDecoder:");
        let t0 = std::time::Instant::now();
        let dict_decoder = tokenizer.to_dictionary_decoder();
        let t1 = std::time::Instant::now();
        let training_duration = t1.duration_since(t0);
        println!("- training_duration: {:#?}", training_duration);
        println!("- size_estimate: {:#?}", dict_decoder.size_estimate());

        println!("training CorpusDecoder:");
        let t0 = std::time::Instant::now();
        let corpus_decoder = tokenizer.to_corpus_decoder();
        let t1 = std::time::Instant::now();
        let training_duration = t1.duration_since(t0);
        println!("- training_duration: {:#?}", training_duration);
        println!("- size_estimate: {:#?}", corpus_decoder.size_estimate());

        if args.time_encode_decode {
            // TODO: `indicatif` for optional progress bar for users waiting on this.

            println!("timing encode/decode:");
            let mut sample_sizes = Vec::new();
            let mut encode_durations = Vec::new();

            // Read the first batch of the first shard for timing.
            let batch = cache.read_cached_batches(shards[0])?.next().unwrap()?;
            let column = batch
                .column_by_name("text")
                .expect("failed to find 'text' column in batch")
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();

            let mut token_groups: Vec<Vec<T>> = Vec::new();

            for sample in column.iter() {
                let sample = sample.unwrap().to_string();

                sample_sizes.push(sample.len());

                let t0 = std::time::Instant::now();
                token_groups.push(tokenizer.encode::<&str>(&sample));
                let t1 = std::time::Instant::now();
                let duration = t1.duration_since(t0);
                encode_durations.push(duration);
            }
            let count = encode_durations.len();
            println!("encode/decode sample count: {}", count);
            let avg_sample_size = sample_sizes.iter().sum::<usize>() / count;
            println!("avg sample size: {}", avg_sample_size);

            let encode_avg = Duration::from_nanos(
                encode_durations
                    .into_iter()
                    .map(|d| d.as_nanos() as u64 / count as u64)
                    .sum::<u64>(),
            );
            println!("tokenizer.encode avg duration: {:#?}", encode_avg);

            // Batch the tokens separately to try and get some cache-locality.
            let mut decode_durations = Vec::new();
            for tokens in &token_groups {
                let t0 = std::time::Instant::now();
                let _ = dict_decoder.decode_to_string(tokens);
                let t1 = std::time::Instant::now();
                let duration = t1.duration_since(t0);
                decode_durations.push(duration);
            }
            let decode_avg = Duration::from_nanos(
                decode_durations
                    .into_iter()
                    .map(|d| d.as_nanos() as u64 / count as u64)
                    .sum::<u64>(),
            );
            println!(
                "DictionaryDecoder decode_to_string avg duration: {:#?}",
                decode_avg
            );

            // Batch the tokens separately to try and get some cache-locality.
            let mut decode_durations = Vec::new();
            for tokens in &token_groups {
                let t0 = std::time::Instant::now();
                let _ = corpus_decoder.decode_to_string(tokens);
                let t1 = std::time::Instant::now();
                let duration = t1.duration_since(t0);
                decode_durations.push(duration);
            }
            let decode_avg = Duration::from_nanos(
                decode_durations
                    .into_iter()
                    .map(|d| d.as_nanos() as u64 / count as u64)
                    .sum::<u64>(),
            );
            println!(
                "CorpusDecoder decode_to_string avg duration: {:#?}",
                decode_avg
            );
        }
    }

    Ok(())
}
