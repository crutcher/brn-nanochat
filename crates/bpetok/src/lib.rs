//! # BPE Tokenizer
#![warn(missing_docs, unused)]

mod merge_job;
mod pair_index;
mod token_types;
mod tokenizer;
mod word;
mod word_count;

pub use merge_job::*;
pub use pair_index::*;
pub use token_types::*;
pub use tokenizer::*;
pub use word::*;
pub use word_count::*;

/// The default value for [`PairIndexOptions::parallel`].
#[cfg(feature = "rayon")]
pub const DEFAULT_PARALLEL: bool = true;
#[cfg(not(feature = "rayon"))]
pub const DEFAULT_PARALLEL: bool = false;
