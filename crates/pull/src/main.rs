use arrow::array::StringArray;
use bpetok::{Tokenizer, TokenizerOptions};
use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use compact_str::CompactString;
use nanochat_data::dataset::DatasetCacheConfig;
use std::collections::HashSet;

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

        let download = true;
        let samples = shards.iter().flat_map(|&shard| {
            cache
                .read_batches(shard, download)
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
        println!("vocab_size: {:#?}", tokenizer.vocab_size());
    }

    /*
    let download = true;
    let mut it = cache.read_batches(0, download)?;
    let batch = it.next().unwrap()?;
    println!("{:#?}", batch);
     */

    Ok(())
}
