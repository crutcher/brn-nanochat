//! # Word Map ``{ Vec<u8> -> T }`` Token Vocabulary

use crate::decoders::pair_decoder::PairExpansionDecoder;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{ByteSpanTokenMap, TokenType};
use crate::vocab::byte_table::ByteTable;
use crate::vocab::pair_vocab::PairTokenMapVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: TokenType", deserialize = "T: TokenType"))]
pub struct ByteSpanTokenMapVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    pub span_map: ByteSpanTokenMap<T>,
}

impl<T: TokenType> Default for ByteSpanTokenMapVocab<T> {
    fn default() -> Self {
        ByteSpanTokenMapVocab::from_byte_table(&ByteTable::default())
    }
}

impl<T: TokenType> From<ByteSpanTokenMap<T>> for ByteSpanTokenMapVocab<T> {
    fn from(words: ByteSpanTokenMap<T>) -> Self {
        Self { span_map: words }
    }
}

impl<'a, T: TokenType> IntoIterator for &'a ByteSpanTokenMapVocab<T> {
    type Item = (&'a Vec<u8>, &'a T);

    type IntoIter = std::collections::hash_map::Iter<'a, Vec<u8>, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.span_map.iter()
    }
}

impl<T: TokenType> ByteSpanTokenMapVocab<T> {
    /// Create a word vocabulary from a byte/token mapping table.
    pub fn from_byte_table(byte_table: &ByteTable<T>) -> Self {
        let mut words = ByteSpanTokenMap::default();
        for idx in 0..256 {
            let byte = idx as u8;
            let token = byte_table.get_token(byte);
            words.insert(vec![byte], token);
        }

        Self { span_map: words }
    }
    /// Shrinks the capacity of the underlying data structures to fit its current size.
    pub fn shrink_to_fit(&mut self) {
        self.span_map.shrink_to_fit();
    }

    /// The number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.span_map.len()
    }

    /// Returns `true` if the vocabulary contains no words.
    pub fn is_empty(&self) -> bool {
        self.span_map.is_empty()
    }

    /// Iterate over the words in the vocabulary.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a Vec<u8>, &'a T)> + 'a {
        self.span_map.iter()
    }

    /// Add a word to the vocab.
    pub fn add_str_word(
        &mut self,
        word: &str,
        token: T,
    ) {
        self.add_bytes_word(word.as_bytes().to_vec(), token);
    }

    /// Add a word to the vocab.
    pub fn add_bytes_word(
        &mut self,
        word: Vec<u8>,
        token: T,
    ) {
        self.span_map.insert(word, token);
    }

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        self.span_map.get(chunk).copied()
    }

    /// Build word vocabulary from a [`PairTokenMapVocab<T>`].
    pub fn from_pair_vocab(pair_vocab: &PairTokenMapVocab<T>) -> Self {
        let mut vocab = Self::default();
        vocab.extend_from_pair_vocab(pair_vocab, true);
        vocab
    }

    /// Extend the word vocabulary from a BPE map vocabulary.
    ///
    /// # Arguments
    /// * `pair_vocab` - the source pair vocab.
    /// * `overwrite` - whether to overwrite existing entries in the word vocab.
    pub fn extend_from_pair_vocab(
        &mut self,
        pair_vocab: &PairTokenMapVocab<T>,
        overwrite: bool,
    ) {
        let skip: Option<AHashSet<T>> = if overwrite {
            None
        } else {
            Some(self.unordered_tokens_iter().collect())
        };

        let decoder = PairExpansionDecoder::from_pair_map(&pair_vocab.pairs);
        for token in pair_vocab.unordered_tokens_iter() {
            if let Some(skip) = &skip
                && skip.contains(&token)
            {
                continue;
            }

            let tokens = [token];
            let chunk = decoder.try_decode_to_bytes(tokens).unwrap();
            self.add_bytes_word(chunk, token);
        }
    }

    /// Build a binary pair map from the word vocabulary.
    pub fn to_pair_vocab(&self) -> PairTokenMapVocab<T> {
        let mut pair_vocab = PairTokenMapVocab::<T>::default();

        let token_to_words: AHashMap<T, &[u8]> = self
            .span_map
            .iter()
            .map(|(chunk, &token)| (token, chunk.as_ref()))
            .collect();

        for token in self.unordered_tokens_iter() {
            let word = token_to_words[&token];

            let k = word.len();
            for p in 1..k {
                if let Some(a) = self.lookup_token(&word[..p])
                    && let Some(b) = self.lookup_token(&word[p..])
                    && a < token
                    && b < token
                {
                    pair_vocab.add_pair((a, b), token);
                }
            }
        }

        pair_vocab
    }
}

impl<T: TokenType> From<&PairTokenMapVocab<T>> for ByteSpanTokenMapVocab<T> {
    fn from(pair_vocab: &PairTokenMapVocab<T>) -> Self {
        Self::from_pair_vocab(pair_vocab)
    }
}

impl<T: TokenType> From<&ByteSpanTokenMapVocab<T>> for PairTokenMapVocab<T> {
    fn from(vocab: &ByteSpanTokenMapVocab<T>) -> Self {
        vocab.to_pair_vocab()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for ByteSpanTokenMapVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }

    fn max_token(&self) -> T {
        self.span_map
            .values()
            .max()
            .copied()
            .unwrap_or(T::from_u8(u8::MAX).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokens_iter() {
        type T = u32;

        let byte_table: ByteTable<T> = Default::default();

        let mut vocab = ByteSpanTokenMapVocab::<T>::default();

        assert_eq!(vocab.max_token(), byte_table.max_token());
        assert_eq!(&vocab.sorted_tokens(), &byte_table.sorted_tokens());

        vocab.add_str_word("apple", 300);
        vocab.add_str_word("banana", 301);
        vocab.add_str_word("pear", 302);

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

    #[test]
    fn test_lookup_token() {
        type T = u32;
        let mut vocab = ByteSpanTokenMapVocab::<T>::default();
        vocab.add_str_word("apple", 300);
        vocab.add_str_word("a", 301);

        assert_eq!(vocab.lookup_token(b"apple"), Some(300));
        assert_eq!(vocab.lookup_token(b"a"), Some(301));
        assert_eq!(vocab.lookup_token(b"b"), Some('b' as u32));
    }

    #[test]
    fn test_build_pair_vocab() {
        type T = u32;
        let mut vocab = ByteSpanTokenMapVocab::<T>::default();
        vocab.add_str_word("at", 300);
        vocab.add_str_word("ate", 301);
        vocab.add_str_word("cat", 302);

        let pair_vocab = vocab.to_pair_vocab();
        assert_eq!(
            &pair_vocab.pairs,
            &[
                (('a' as u32, 't' as u32), 300),
                ((300, 'e' as u32), 301),
                (('c' as u32, 300), 302)
            ]
            .iter()
            .map(|&(a, b)| (a, b))
            .collect::<AHashMap<_, _>>()
        );
    }
}
