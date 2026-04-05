use std::{
    collections::HashSet,
    sync::Arc,
};

use burn::tensor::{
    AsIndex,
    Slice,
};
use clap::Parser;
use llm_dataloader::{
    reader::{
        TokenBatchIteratorOptions,
        compact_dense_token_blocks,
        select_text_columns,
        tokenize_text_batches,
    },
    support::arrow::parquet_shards::read_parquet_shards,
};
use nanochat_data::dataset::DatasetCacheConfig;
use wordchipper::{
    Tokenizer,
    UnifiedTokenVocab,
    VocabIndex,
    disk_cache::WordchipperDiskCache,
};
use wordchipper_cli_util::logging::LogArgs;

#[derive(Debug, Clone, clap::Args)]
pub struct TokenBatchOptionsArgs {
    /// The number of sequences to load per batch.
    #[arg(long, default_value_t = 32)]
    pub batch_size: usize,

    /// The maximum number of tokens in a sequence.
    #[arg(long, default_value_t = 2048)]
    pub batch_seq_len: usize,

    /// The minimum number of sequences to keep in the buffer
    /// before loading more sequences.
    #[arg(long, default_value_t = 1024)]
    pub min_buffer: usize,
}

impl TokenBatchOptionsArgs {
    pub fn options(&self) -> TokenBatchIteratorOptions {
        TokenBatchIteratorOptions {
            batch_size: self.batch_size,
            batch_seq_len: self.batch_seq_len,
            min_buffer: self.min_buffer,
        }
    }
}

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

    #[command(flatten)]
    pub token_batch_options: TokenBatchOptionsArgs,

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

    // Iterator<ArrowResult<RecordBatch>>
    let parquet_batches = read_parquet_shards(shard_paths);

    // Iterator<ArrowResult<Vec<String>>>
    let sample_batches = select_text_columns("text", parquet_batches);

    // Iterator<ArrowResult<Vec<Vec<u32>>>>
    let token_batches = tokenize_text_batches(tok.clone(), sample_batches);

    // Iterator<ArrowResult<Vec<Vec<u32>>>> (batch_size x batch_seq_len)
    let dense_blocks = compact_dense_token_blocks(
        args.token_batch_options.options(),
        vec![bos_token],
        vec![],
        token_batches,
    );

    for (idx, res) in dense_blocks.enumerate() {
        let block = res?;

        let b = block.len();
        let k = block.first().unwrap().len();

        assert_eq!(b, args.token_batch_options.batch_size);
        block.iter().for_each(|seq| {
            assert_eq!(seq.len(), args.token_batch_options.batch_seq_len);
        });

        println!("{idx}: {b} x {k}");
    }

    Ok(())
}
