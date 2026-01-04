//! # Tokenizer Data

use crate::{MergeMap, Pair, TokenType};

/// Core data describing a BPE Tokenizer.
#[derive(Debug, Clone)]
pub struct TokenizerData<T: TokenType> {
    /// Maps [`Pair<T>`] to [`T`], representing the byte pair encoding merges.
    pub merge_map: MergeMap<T>,

    /// The regex pattern used for text splitting.
    pub pattern: String,
}

impl<T: TokenType> TokenizerData<T> {
    /// Size estimate in bytes.
    pub fn size_estimate(&self) -> usize {
        self.merge_map.capacity() * std::mem::size_of::<Pair<T>>() + self.pattern.len()
    }

    /// Returns an iterator over the non-byte tokens in this map.
    pub fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.merge_map.values().copied()
    }

    /// Gets the highest ranked token.
    pub fn max_token(&self) -> T {
        self.pair_tokens().max().unwrap()
    }
}
