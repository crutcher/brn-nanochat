use burn::tensor::Slice;
use clap::Parser;

pub mod dataset;
/// Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Indices to load.
    #[arg(short, long, value_delimiter = ',', default_value = "0")]
    pub indices: Vec<Slice>,
}

fn main() {
    let args = Args::parse();
    println!("{:#?}", args);

    let _cache = dataset::DatasetCache::default();
}
