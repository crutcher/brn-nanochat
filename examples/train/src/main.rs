use std::{
    collections::HashSet,
    sync::Arc,
};

use burn::{
    nn::{
        Embedding,
        EmbeddingConfig,
    },
    tensor::{
        AsIndex,
        Slice,
        backend::AutodiffBackend,
    },
};
use clap::Parser;
use llm_dataloader::reader::TokenBatchIteratorOptions;
use wordchipper::{
    UnifiedTokenVocab,
    VocabIndex,
    disk_cache::WordchipperDiskCache,
};
use wordchipper_cli_util::logging::LogArgs;
use zsl_chat::gpt::gpt_model::GPTConfig;
use zsl_data_cache::dataset::DatasetCacheConfig;

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

#[derive(Parser, Debug)]
pub struct Args {
    #[clap(flatten)]
    pub logging: LogArgs,

    /// The embedding dimension size.
    #[clap(long, default_value = "768")]
    pub embedding_dim: usize,

    /// The pretrained vocabulary.
    #[clap(long, default_value = "openai:p50k_edit")]
    pub pretrained_vocab: String,

    /// Beginning of sequence token.
    #[arg(long, default_value = "<|bos|>")]
    pub bos_token: String,

    /// Shards to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub shards: Vec<Slice>,

    /// Path to dataset directory.
    #[arg(long)]
    pub dataset_dir: String,

    #[command(flatten)]
    pub token_batch_options: TokenBatchOptionsArgs,
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.logging.setup_logging(3).unwrap();

    run::<burn::backend::Autodiff<burn::backend::cuda::Cuda>>(&args)
}

fn run<B: AutodiffBackend>(args: &Args) -> anyhow::Result<()> {
    type T = u32;

    println!("{:#?}", args);

    let device: B::Device = Default::default();

    let cache_config = DatasetCacheConfig::new().with_cache_dir(args.dataset_dir.clone());
    log::info!("DATASET CACHE: {:#?}", cache_config);
    let mut cache = cache_config.clone().init()?;

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

    log::info!("Loading Shards: {shards:?}");
    let _shard_paths = cache.load_shards(&shards)?;

    let mut disk_cache = WordchipperDiskCache::default();
    let mut vocab: UnifiedTokenVocab<T> =
        wordchipper::load_vocab(&args.pretrained_vocab, &mut disk_cache)?
            .vocab()
            .to_token_type()?;

    let vocab_size = vocab.len();

    let max_token = vocab.max_token().unwrap();

    let _bos_token: T = {
        let specials = vocab.special_vocab_mut();
        if let Some(tok) = specials.lookup_token(args.bos_token.as_bytes()) {
            tok
        } else {
            let tok = max_token + 1;
            specials.add_str_word(&args.bos_token, tok);
            tok
        }
    };
    let vocab = Arc::new(vocab);

    let _tok = wordchipper::TokenizerOptions::default()
        .with_accelerated_lexers(true)
        .with_parallel(true)
        .build(vocab);

    let ec = EmbeddingConfig::new(vocab_size, args.embedding_dim);

    let _embedding: Embedding<B> = ec.init::<B>(&device);

    let gpt_config = GPTConfig::new().with_vocab_size(vocab_size);

    let _gpt = gpt_config.init::<B>(&device);

    Ok(())
}
