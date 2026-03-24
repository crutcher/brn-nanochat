use burn::nn::{Embedding, EmbeddingConfig};
use burn::tensor::backend::AutodiffBackend;
use clap::Parser;
use nanochat::gpt::gpt_model::GPTConfig;
use wordchipper::VocabIndex;
use wordchipper::disk_cache::WordchipperDiskCache;
use wordchipper_cli_util::logging::LogArgs;

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
}

#[cfg(feature = "cuda")]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    args.logging.setup_logging(3).unwrap();

    run::<burn::backend::Autodiff<burn::backend::cuda::Cuda>>(&args)
}

fn run<B: AutodiffBackend>(args: &Args) -> anyhow::Result<()> {
    println!("{:#?}", args);

    let device: B::Device = Default::default();

    let mut disk_cache = WordchipperDiskCache::default();
    let vocab = wordchipper::load_vocab(&args.pretrained_vocab, &mut disk_cache)?
        .vocab()
        .clone();

    let vocab_size = vocab.len();

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
