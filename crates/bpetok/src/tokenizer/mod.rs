//! # Tokenizer Structures

pub mod cps_tokenizer;

use crate::types::TokenType;
use crate::vocab::data::TokenVocabData;
pub use cps_tokenizer::*;
use std::sync::Arc;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType> {
    /// Returns a reference to the core data describing this tokenizer.
    fn data(&self) -> &Arc<TokenVocabData<T>>;

    /// Returns an iterator over the non-byte tokens in this map.
    fn pair_tokens(&self) -> impl Iterator<Item = T>;

    /// Returns the maximum token id in this decoder.
    fn max_token(&self) -> T {
        self.pair_tokens().max().unwrap()
    }

    /// Encode text into tokens.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T>;
}
