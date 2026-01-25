//! # Word Map ``{ Vec<u8> -> T }`` Token Vocabulary

use crate::types::{ByteSpanTokenMap, TokenType};
use crate::vocab::{ByteTokenTable, PairTokenMapVocab, TokenVocabIndex};
use ahash::AHashMap;
use alloc::sync::Arc;

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteSpanTokenMapVocab<T: TokenType> {
    /// The byte/token mapping table.
    byte_table: Arc<ByteTokenTable<T>>,

    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    span_map: ByteSpanTokenMap<T>,
}

impl<T: TokenType> Default for ByteSpanTokenMapVocab<T> {
    fn default() -> Self {
        ByteSpanTokenMapVocab::from_byte_table(Arc::new(ByteTokenTable::default()))
    }
}

impl<T: TokenType> From<ByteSpanTokenMap<T>> for ByteSpanTokenMapVocab<T> {
    fn from(span_map: ByteSpanTokenMap<T>) -> Self {
        Self::from_span_map(span_map)
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
    /// Build vocabulary from just a [`ByteTokenTable`].
    ///
    /// Will have 255 span entries, each 1-byte long.
    pub fn from_byte_table<B>(byte_table: B) -> Self
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        let byte_table = byte_table.into();

        let span_map: ByteSpanTokenMap<T> = byte_table.to_span_pairs().collect();

        Self::new(byte_table, span_map)
    }

    /// Build a vocabulary from just a [`ByteSpanTokenMap`].
    ///
    /// The [`ByteTokenTable`] will be inferred from the [`ByteSpanTokenMap`],
    /// and the default ordinal byte to token mappings.
    ///
    /// # Panics
    /// If the [`ByteTokenTable`] mapping is not 1:1.
    pub fn from_span_map(span_map: ByteSpanTokenMap<T>) -> Self {
        let byte_to_token = (0..256)
            .map(|ord| {
                let byte = ord as u8;
                if let Some(&token) = span_map.get(&vec![byte]) {
                    token
                } else {
                    T::from_usize(ord).unwrap()
                }
            })
            .collect::<Vec<_>>();

        let byte_table = Arc::new(ByteTokenTable::from_byte_to_token(&byte_to_token));

        Self::new(byte_table, span_map)
    }

    /// Build word vocabulary from a [`PairTokenMapVocab<T>`].
    pub fn from_pair_vocab(pair_vocab: &PairTokenMapVocab<T>) -> Self {
        let byte_table: Arc<ByteTokenTable<T>> = pair_vocab.byte_table().clone();
        let span_map: ByteSpanTokenMap<T> = byte_table.to_span_pairs().collect();
        Self::new(byte_table, span_map)
    }

    /// Build vocabulary.
    ///
    /// The span map will be the union of the span map,
    /// and all overrides from the `byte_table`.
    ///
    /// # Panics
    /// If the [`ByteTokenTable`] disagrees with the `span_map`.
    pub fn new<B>(
        byte_table: B,
        mut span_map: ByteSpanTokenMap<T>,
    ) -> Self
    where
        B: Into<Arc<ByteTokenTable<T>>>,
    {
        let byte_table = byte_table.into();
        for (span, token) in byte_table.to_span_pairs() {
            if let Some(previous) = span_map.insert(span, token)
                && previous != token
            {
                panic!(
                    "ByteTable disagrees with span_map: {:?} != {:?}",
                    previous, token
                );
            }
        }

        Self {
            byte_table,
            span_map,
        }
    }

    /// Get the span => token map.
    pub fn span_map(&self) -> &ByteSpanTokenMap<T> {
        &self.span_map
    }

    /// Iterate over the span => token pairs.
    pub fn to_span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
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

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        self.span_map.get(chunk).copied()
    }

    /// Build a binary pair map from the word vocabulary.
    pub fn to_pair_vocab(&self) -> PairTokenMapVocab<T> {
        let byte_table = self.byte_table.clone();

        let mut pairs = AHashMap::default();

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
                    pairs.insert((a, b), token);
                }
            }
        }

        PairTokenMapVocab::new(byte_table, pairs)
    }
}

impl<T: TokenType> TokenVocabIndex<T> for ByteSpanTokenMapVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::{ByteTokenTable, TokenVocabIndex};

    #[test]
    fn test_tokens_iter() {
        type T = u32;

        let byte_table: ByteTokenTable<T> = Default::default();

        let vocab = ByteSpanTokenMapVocab::<T>::default();

        assert_eq!(vocab.max_token(), byte_table.max_token());
        assert_eq!(&vocab.sorted_tokens(), &byte_table.sorted_tokens());

        let mut span_map = vocab.span_map().clone();

        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("banana".as_bytes().to_vec(), 301);
        span_map.insert("pear".as_bytes().to_vec(), 302);

        let vocab = ByteSpanTokenMapVocab::from(span_map);

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

        let mut span_map: AHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("apple".as_bytes().to_vec(), 300);
        span_map.insert("a".as_bytes().to_vec(), 301);

        let vocab = ByteSpanTokenMapVocab::<T>::from_span_map(span_map);

        assert_eq!(vocab.lookup_token(b"apple"), Some(300));
        assert_eq!(vocab.lookup_token(b"a"), Some(301));
        assert_eq!(vocab.lookup_token(b"b"), Some('b' as u32));
    }

    #[test]
    fn test_build_pair_vocab() {
        type T = u32;

        let mut span_map: AHashMap<Vec<u8>, T> = Default::default();
        span_map.insert("at".as_bytes().to_vec(), 300);
        span_map.insert("ate".as_bytes().to_vec(), 301);
        span_map.insert("cat".as_bytes().to_vec(), 302);

        let vocab = ByteSpanTokenMapVocab::from(span_map);

        let pair_vocab = vocab.to_pair_vocab();
        assert_eq!(
            pair_vocab.pairs(),
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
