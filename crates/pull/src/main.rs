use burn::tensor::Slice;
use clap::{Parser, arg};

pub mod slice_parser;

use slice_parser::parse_slice;

/// Nanochat Data Loader.
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
pub struct Args {
    /// Indices to load.
    #[arg(short, long, value_parser = parse_slice, value_delimiter = ',', default_value = "0")]
    pub indices: Vec<Slice>,
}

fn main() {
    let args = Args::parse();

    println!("{:#?}", args);
}
