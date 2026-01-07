//! # BPE Tokenizer
#![warn(missing_docs, unused)]

pub mod decoder;
pub mod tokenizer;
pub mod types;
pub mod util;
pub mod validators;
pub mod vocab;

/// Default GPT-4 style regex pattern for splitting text
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Default regex pattern for splitting text.
pub const DEFAULT_PATTERN: &str = GPT4_PATTERN;

/// The default value for [`PairIndexOptions::parallel`].
#[cfg(feature = "rayon")]
pub const DEFAULT_PARALLEL: bool = true;
#[cfg(not(feature = "rayon"))]
pub const DEFAULT_PARALLEL: bool = false;
