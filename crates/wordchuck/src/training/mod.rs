//! # Vocabulary Training

pub mod pair_span_index;
pub mod text_span_counter;
pub mod token_span_buffer;
pub mod trainer;

pub use trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
