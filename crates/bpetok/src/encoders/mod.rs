//! # Tokenizer Structures

pub mod unified_encoder;

use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
pub use unified_encoder::*;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType>: TokenVocabIndex<T> + Send + Sync {
    /// Encode a word, and append the resulting tokens to the given token buffer.
    fn encode_append_word(
        &self,
        word: &str,
        tokens: &mut Vec<T>,
    );

    /// Encode bytes into tokens.
    fn encode_append(
        &self,
        source: &str,
        tokens: &mut Vec<T>,
    );

    /// Encode text into tokens.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());
        self.encode_append(text, &mut tokens);
        tokens
    }

    /// Encode a batch of text into tokens.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        batch.iter().map(|s| self.encode(s)).collect()
    }
}
