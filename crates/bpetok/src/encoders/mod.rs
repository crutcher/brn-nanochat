//! # Tokenizer Structures

pub mod unified_encoder;

use crate::types::TokenType;
pub use unified_encoder::*;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType> {
    /// Returns the maximum token id in this decoders.
    fn max_token(&self) -> T;

    /// Encode text into tokens.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T>;

    /// Encode a batch of text into tokens.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        batch.iter().map(|s| self.encode(s)).collect()
    }
}
