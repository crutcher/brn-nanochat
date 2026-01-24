//! # Token Vocabulary Index

use crate::types::TokenType;

/// Returns an iterator over all byte tokens (0-255).
pub fn byte_tokens_iter<T: TokenType>() -> impl Iterator<Item = T> {
    (0..=255).map(move |i| T::from_usize(i).unwrap())
}

/// Common traits for token vocabularies.
pub trait TokenVocabIndex<T: TokenType>: Clone + Send + Sync {
    /// Returns an iterator over all non-byte tokens.
    ///
    /// All returned tokens will have rank >= 256.
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T>;

    /// Returns a sorted vector of all tokens.
    fn sorted_tokens(&self) -> Vec<T> {
        let mut tokens: Vec<T> = self.unordered_tokens_iter().collect();
        tokens.sort();
        tokens
    }

    /// Gets the highest ranked token.
    fn max_token(&self) -> T {
        self.unordered_tokens_iter().max().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use crate::vocab::vocab_index::byte_tokens_iter;
    #[test]
    fn test_byte_tokens_iter() {
        assert_eq!(
            byte_tokens_iter::<u32>().collect::<Vec<_>>(),
            (0_u32..=255).collect::<Vec<_>>()
        );
    }
}
