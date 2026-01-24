//! # Pair Map ``{ (T, T) -> T }`` Token Vocabulary

use crate::types::{PairTokenMap, TokenType};
use crate::vocab::byte_table::ByteTable;
use crate::vocab::vocab_index::TokenVocabIndex;

/// Token vocabulary as a binary-pair encoding map of ``{ (T, T) -> T }``.
#[derive(Default, Debug, Clone)]
pub struct PairTokenMapVocab<T: TokenType> {
    /// Byte/token mapping table.
    pub byte_table: ByteTable<T>,

    /// Map of ``{ (T, T) -> T }``.
    pub pairs: PairTokenMap<T>,
}

impl<'a, T: TokenType> IntoIterator for &'a PairTokenMapVocab<T> {
    type Item = (&'a (T, T), &'a T);
    type IntoIter = std::collections::hash_map::Iter<'a, (T, T), T>;
    fn into_iter(self) -> Self::IntoIter {
        self.pairs.iter()
    }
}

impl<T: TokenType> PairTokenMapVocab<T> {
    /// The number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Returns `true` if the vocabulary contains no words.
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }

    /// Iterate over the pairs in the vocabulary.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a (T, T), &'a T)> + 'a {
        self.pairs.iter()
    }

    /// Add a pair to the vocab.
    pub fn add_pair(
        &mut self,
        pair: (T, T),
        token: T,
    ) {
        self.pairs.insert(pair, token);
    }

    /// Shrinks the capacity of the underlying data structures to fit its current size.
    pub fn shrink_to_fit(&mut self) {
        self.pairs.shrink_to_fit();
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
