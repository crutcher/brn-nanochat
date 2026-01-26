//! # Token Vocabulary Index

use crate::types::TokenType;

/// Common traits for token vocabularies.
pub trait TokenVocab<T: TokenType>: Clone + Send + Sync {
    /// Returns an iterator over all non-byte tokens.
    ///
    /// All returned tokens will have rank >= 256.
    fn unordered_tokens(&self) -> impl Iterator<Item = T>;

    /// Returns a sorted vector of all tokens.
    fn sorted_tokens(&self) -> Vec<T> {
        let mut tokens: Vec<T> = self.unordered_tokens().collect();
        tokens.sort();
        tokens
    }

    /// Gets the highest ranked token.
    fn max_token(&self) -> T {
        self.unordered_tokens().max().unwrap()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    fn span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)>;
}
