//! # Token Encoders
//!
//! ## Examples
//!
//! ```rust,no_run
//! use wordchuck::vocab::UnifiedTokenVocab;
//! use wordchuck::encoders::MergeHeapVocabEncoder;
//! use wordchuck::types::TokenType;
//! use std::sync::Arc;
//!
//! fn example<T: TokenType>(
//!     vocab: Arc<UnifiedTokenVocab<T>>,
//!     batch: &[String],
//! ) -> Vec<Vec<T>> {
//!     let encoder: MergeHeapVocabEncoder<T> = MergeHeapVocabEncoder::init(vocab);
//!
//!     #[cfg(feature = "rayon")]
//!     let encoder = wordchuck::rayon::RayonEncoder::new(encoder);
//!
//!     encoder.encode_batch(batch);
//! }
//! ```

pub mod merge_heap_encoder;
pub mod token_encoder;

pub use merge_heap_encoder::MergeHeapVocabEncoder;
pub use token_encoder::TokenEncoder;
