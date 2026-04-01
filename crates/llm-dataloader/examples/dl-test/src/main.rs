use burn::tensor::{AsIndex, Slice};
use clap::Parser;
use llm_dataloader::loader::TokenBatchIterator;
use nanochat_data::dataset::DatasetCacheConfig;
use std::collections::HashSet;
use std::sync::Arc;
use wordchipper::disk_cache::WordchipperDiskCache;
use wordchipper::{Tokenizer, UnifiedTokenVocab, VocabIndex};
use wordchipper_cli_util::logging::LogArgs;

/// Example Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    /// The vocab model to use.
    #[arg(long, default_value = "openai:p50k_base")]
    pub vocab_model: String,

    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    /// Logging configuration.
    #[clap(flatten)]
    logging: LogArgs,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    args.logging.setup_logging(3).unwrap();

    log::info!("ARGS: {:#?}", args);

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir);
    log::info!("DATASET CACHE: {:#?}", cache_config);

    type T = u32;

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

    log::info!("Loading Shards: {shards:?}");
    let shard_paths = cache.load_shards(&shards)?;

    let mut wc_disk_cache: WordchipperDiskCache = Default::default();

    log::info!("Loading Vocab: {:?}", args.vocab_model);
    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.vocab_model, &mut wc_disk_cache)?
            .vocab()
            .to_token_type()?;

    let max_token = vocab.max_token().unwrap();

    let bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            tok
        }
    };

    let tok: Arc<Tokenizer<T>> = wordchipper::TokenizerOptions::default()
        .with_parallel(true)
        .with_accelerated_lexers(true)
        .build(vocab.into());

    let mut it: TokenBatchIterator<T> =
        TokenBatchIterator::new(tok, shard_paths, Default::default(), bos_token);

    let mut idx = 0;
    while let Some(batch) = it.next_batch()? {
        let total = batch.total_tokens();
        log::info!("{idx}: {:?}", total);
        idx += 1;
    }

    Ok(())
}
