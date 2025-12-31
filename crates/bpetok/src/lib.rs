//! # BPE Tokenizer
#![warn(missing_docs, unused)]

mod decoder;
mod merge_job;
mod pair_index;
mod token_types;
mod tokenizer;
mod word;
mod word_count;

pub use decoder::*;
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

/// Default GPT-4 style regex pattern for splitting text
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Default regex pattern for splitting text.
pub const DEFAULT_PATTERN: &str = GPT4_PATTERN;

/// Default number of reserved tokens
pub const DEFAULT_NUM_RESERVED: usize = 256;
