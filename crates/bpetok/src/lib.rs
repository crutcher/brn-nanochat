//! # BPE Tokenizer
#![warn(missing_docs, unused)]

mod merge_job;
mod pair_index;
mod split;
mod token_types;
mod tokenizer;
mod word;

pub use merge_job::*;
pub use pair_index::*;
pub use split::*;
pub use token_types::*;
pub use tokenizer::*;
pub use word::*;

/// The default value for [`PairIndexOptions::parallel`].
#[cfg(feature = "rayon")]
pub const DEFAULT_PARALLEL: bool = true;
#[cfg(not(feature = "rayon"))]
pub const DEFAULT_PARALLEL: bool = false;
