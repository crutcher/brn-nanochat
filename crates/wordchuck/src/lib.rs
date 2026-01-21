//! # LLM Tokenizer
//!
//! Work in Progress.
//!
//! # Training Example
//!
//! Consider the following, to train a tokenizer and export it a "*.tiktoken" file.
//!
//! - the iterator stream for samples may be quite large.
//! - training a `nanochat` equivalent tokenizer takes ~80 CPU minutes.
//!
//! ```rust,ignore
//! let options = BinaryPairVocabTrainerOptions::new_with_vocab_size(args.vocab_size);
//!
//! let mut trainer = options.init::<K, C>();
//!
//! for batch in batches {
//!     trainer.update_from_sampples(batch.iter());
//! }
//!
//! let vocab: Arc<UnifiedTokenVocab<T>> = trainer
//!     .train()
//!     .expect("training failed")
//!     .into();
//!
//! if let Some(path) = args.tiktoken_save_path {
//!     vocab.word_vocab.save_to_tiktoken_path(&path)?;
//!     println!("- tiktoken vocab: {path:?}");
//! }
//!
//! let encoder: UnifiedVocabEncoder<T> = UnifiedVocabEncoder::<T>::new(vocab.clone());
//! let encoder = ParallelEncoder::new(encoder);
//!
//! let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
//! let decoder = ParallelDecoder::new(decoder);
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
