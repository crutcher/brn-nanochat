//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::types::{PairTokenMap, TokenType};
use crate::vocab::byte_table::ByteTable;
use crate::vocab::vocab_index::TokenVocabIndex;

/// Pair ``(T, T) -> T`` Vocabulary.
///
/// - Grounded in a `ByteTable<T>` for byte-to-token mapping.
/// - Collection of ``(T, T) -> T`` pairs.
#[derive(Default, Debug, Clone)]
pub struct PairTokenMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    byte_table: ByteTable<T>,

    /// Map of ``{ (T, T) -> T }``.
    pairs: PairTokenMap<T>,
}

impl<T: TokenType> PairTokenMapVocab<T> {
    /// Create a new vocab.
    pub fn new(
        byte_table: &ByteTable<T>,
        pairs: PairTokenMap<T>,
    ) -> Self {
        let byte_table = byte_table.clone();
        Self { byte_table, pairs }
    }

    /// Get the byte/token mapping table.
    pub fn byte_table(&self) -> &ByteTable<T> {
        &self.byte_table
    }

    /// Get the map of pairs.
    pub fn pairs(&self) -> &PairTokenMap<T> {
        &self.pairs
    }
}

impl<T: TokenType> TokenVocabIndex<T> for PairTokenMapVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.byte_table
            .unordered_tokens_iter()
            .chain(self.pairs.values().copied())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_sorted() {
        type T = u32;
        let byte_table: ByteTable<T> = Default::default();

        let mut vocab = PairTokenMapVocab::<T> {
            pairs: PairTokenMap::default(),
            byte_table: byte_table.clone(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(&vocab.sorted_tokens(), &byte_table.sorted_tokens());

        vocab.pairs.insert((1, 2), 300);
        vocab.pairs.insert((3, 4), 301);
        vocab.pairs.insert((300, 301), 302);

        assert_eq!(vocab.max_token(), 302);

        assert_eq!(
            &vocab.sorted_tokens(),
            &byte_table
                .sorted_tokens()
                .into_iter()
                .chain([300_u32, 301, 302].into_iter())
                .collect::<Vec<T>>()
        );
    }
}
