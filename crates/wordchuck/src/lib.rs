//! # `WordChuck` LLM Tokenizer
//!
//!
//! ## Training Example
//!
//! Consider the following, to train a tokenizer and export it a "*.tiktoken" file.
//!
//! - the iterator stream for samples may be quite large.
//! - training a `nanochat` equivalent tokenizer takes ~80 CPU minutes.
//!
//! ```rust,no_run
//! use wordchuck::training::bpe_trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
//! use wordchuck::vocab::io::tiktoken_io::save_span_map_to_tiktoken_path;
//! use wordchuck::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
//! use wordchuck::vocab::{ByteTokenTable, UnifiedTokenVocab};
//! use wordchuck::encoders::UnifiedVocabEncoder;
//! use wordchuck::decoders::DictionaryDecoder;
//! use wordchuck::rayon::{ParallelRayonEncoder, ParallelRayonDecoder};
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
//!         OA_GPT3_CL100K_WORD_PATTERN,
//!         vocab_size,
//!     );
//!
//!     let mut trainer: BinaryPairVocabTrainer<K, C> = options.init();
//!
//!     for batch in batches {
//!         // The trainer has no parallelism.
//!         // The perceived benefits of parallelism in the trainer
//!         // are insignificant if the IO for the sample source is
//!         // fed by another thread.
//!         trainer.update_from_samples(batch.as_ref());
//!     }
//!
//!     let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());
//!
//!     let vocab: Arc<UnifiedTokenVocab<T>> = trainer
//!         .train(byte_table.clone())
//!         .expect("training failed")
//!         .into();
//!
//!     if let Some(path) = tiktoken_save_path {
//!         save_span_map_to_tiktoken_path(&vocab.word_vocab.span_map(), &path)
//!             .expect("failed to save tiktoken vocab");
//!         println!("- tiktoken vocab: {path:?}");
//!     }
//!
//!     let encoder: UnifiedVocabEncoder<T> = UnifiedVocabEncoder::<T>::new(vocab.clone());
//!     let encoder = ParallelRayonEncoder::new(encoder);
//!
//!     let decoder = DictionaryDecoder::new(vocab.unified_dictionary());
//!     let decoder = ParallelRayonDecoder::new(decoder);
//! }
//! ```
#![warn(missing_docs, unused)]

extern crate alloc;

#[cfg(feature = "rayon")]
pub mod rayon;

pub mod decoders;
pub mod encoders;
pub mod regex;
pub mod segmentation;
pub mod training;
pub mod types;
pub mod util;
pub mod vocab;
