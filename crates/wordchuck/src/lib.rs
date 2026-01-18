//! # LLM Tokenizer
//!
//! Work in Progress.
//!
//! # Training Example
//!
//! Consider the following, to train a tokenizer and export it a "*.tiktoken" file.
//!
//! - the iterator stream for samples may be quite large.
//! - training a `nanochat` equivalent tokenizer takes ~150 CPU minutes.
//!
//! ```rust,ignore
//! let TrainResults::<T> {
//!     word_pattern,
//!     pair_vocab,
//! } = BinaryPairVocabTrainer::new_with_vocab_size(args.vocab_size)
//!     .train_vocab_from_sample_iter::<T, K, C, _>(samples)
//!     .expect("training failed");
//!
//! let vocab: = UnifiedTokenVocab::new(word_pattern.into())
//!     .with_pair_vocab(pair_vocab)
//!     .expand_words_from_bpe();
//!
//! encoder_data.word_vocab.save_to_tiktoken_path(&path)?;
//! ```
#![warn(missing_docs, unused)]

extern crate alloc;

pub mod decoders;
pub mod encoders;
pub mod training;
pub mod types;
pub mod util;
pub mod vocab;

/// Default GPT-4 style regex pattern for splitting text
pub const GPT4_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

/// Default regex pattern for splitting text.
pub const DEFAULT_PATTERN: &str = GPT4_PATTERN;

/// Default value for parallel processing; based on the `rayon` feature.
#[cfg(feature = "rayon")]
pub const DEFAULT_PARALLEL: bool = true;
#[cfg(not(feature = "rayon"))]
pub const DEFAULT_PARALLEL: bool = false;

/// Constant guess for the expected bytes/token ratio.
pub const BYTES_PER_TOKEN_HINT: f64 = 4.0;
