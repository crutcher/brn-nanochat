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
//! ```rust,no_run
//! use wordchuck::training::trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
//! use wordchuck::vocab::io::tiktoken_io::save_word_map_to_tiktoken_path;
//! use wordchuck::vocab::public::patterns::GPT4_PATTERN;
//! use wordchuck::vocab::UnifiedTokenVocab;
//! use wordchuck::encoders::{ParallelRayonEncoder, UnifiedVocabEncoder};
//! use wordchuck::decoders::{ParallelRayonDecoder, DictionaryDecoder};
//! use std::sync::Arc;
//!
//! fn example<I, S>(
//!     vocab_size: usize,
//!     batches: I,
//!     tiktoken_save_path: Option<String>,
//! ) where
//!     I: IntoIterator,
//!     I::Item: AsRef<[S]>,
//!     S: AsRef<str>,
//! {
//!     // We can pick any unsigned integer type > vocab_size;
//!     // See [`wordchuck::types::TokenType`].
//!     type T = u32;
//!     type K = String;
//!     type C = u64;
//!
//!     let options = BinaryPairVocabTrainerOptions::new(
//!         GPT4_PATTERN,
//!         vocab_size,
//!     );
//!
//!     let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();
//!
//!     for batch in batches {
//!         trainer.update_from_samples(batch.as_ref());
//!     }
//!
//!     let vocab: Arc<UnifiedTokenVocab<T>> = trainer
//!         .train::<T>()
//!         .expect("training failed")
//!         .into();
//!
//!     if let Some(path) = tiktoken_save_path {
//!         save_word_map_to_tiktoken_path(&vocab.word_vocab, &path)
//!             .expect("failed to save tiktoken vocab");
//!         println!("- tiktoken vocab: {path:?}");
//!     }
//!
//!     let encoder: UnifiedVocabEncoder<T> = UnifiedVocabEncoder::<T>::new(vocab.clone());
//!     let encoder = ParallelRayonEncoder::new(encoder);
//!
//!     let decoder = DictionaryDecoder::new(vocab.compiled_dictionary());
//!     let decoder = ParallelRayonDecoder::new(decoder);
//! }
//! ```
#![warn(missing_docs, unused)]

extern crate alloc;

pub mod decoders;
pub mod encoders;
pub mod training;
pub mod types;
pub mod util;
pub mod vocab;

/// Default value for parallel processing; based on the `rayon` feature.
#[cfg(feature = "rayon")]
pub const DEFAULT_PARALLEL: bool = true;
#[cfg(not(feature = "rayon"))]
pub const DEFAULT_PARALLEL: bool = false;
