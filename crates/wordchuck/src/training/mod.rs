//! # Vocabulary Training

pub mod bpe_trainer;
pub mod pair_span_index;
pub mod text_span_counter;
pub mod token_span_buffer;

pub use bpe_trainer::{BinaryPairVocabTrainer, BinaryPairVocabTrainerOptions};
