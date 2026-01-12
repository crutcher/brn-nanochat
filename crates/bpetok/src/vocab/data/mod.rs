//! # Vocabulary Data

pub mod pair_vocab;
pub mod unified_vocab;
pub mod word_vocab;

use crate::types::TokenType;
pub use pair_vocab::*;
pub use unified_vocab::*;
pub use word_vocab::*;

/// Returns an iterator over all byte tokens (0-255).
pub fn byte_tokens_iter<T: TokenType>() -> impl Iterator<Item = T> {
    (0..=255).map(move |i| T::from_usize(i).unwrap())
}

/// Common traits for token vocabularies.
pub trait TokenVocab<T: TokenType>: Clone + Send + Sync {
    /// Returns an iterator over all non-byte tokens.
    ///
    /// All returned tokens will have rank >= 256.
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T>;

    /// Returns an iterator over all tokens.
    ///
    /// This will include all byte tokens (0-255),
    /// as well as the tokens returned by [`non_byte_tokens_iter`].
    fn all_tokens_iter(&self) -> impl Iterator<Item = T> {
        byte_tokens_iter().chain(self.compound_tokens_iter())
    }

    /// Gets the highest ranked token.
    fn max_token(&self) -> T;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_tokens_iter() {
        assert_eq!(
            byte_tokens_iter::<u32>().collect::<Vec<_>>(),
            (0_u32..=255).collect::<Vec<_>>()
        );
    }
}
