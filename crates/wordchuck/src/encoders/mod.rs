//! # Tokenizer Structures

pub mod bp_merge;
pub mod text_segmentor;
pub mod token_encoder;
pub mod unified_encoder;

pub use token_encoder::TokenEncoder;
pub use unified_encoder::UnifiedVocabEncoder;

#[cfg(feature = "rayon")]
pub mod rayon_encoder;
#[cfg(feature = "rayon")]
pub use rayon_encoder::ParallelRayonEncoder;
