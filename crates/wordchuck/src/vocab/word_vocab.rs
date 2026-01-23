//! # Word Map ``{ Vec<u8> -> T }`` Token Vocabulary

use crate::decoders::pair_decoder::PairExpansionDecoder;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenType, WordToTokenMap};
use crate::vocab::pair_vocab::PairMapTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::{AHashMap, AHashSet};
use serde::{Deserialize, Serialize};

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: TokenType", deserialize = "T: TokenType"))]
pub struct WordMapTokenVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ &[u8] -> T }``.
    pub words: WordToTokenMap<T>,
}

impl<T: TokenType> From<WordToTokenMap<T>> for WordMapTokenVocab<T> {
    fn from(words: WordToTokenMap<T>) -> Self {
        Self { words }
    }
}

impl<'a, T: TokenType> IntoIterator for &'a WordMapTokenVocab<T> {
    type Item = (&'a Vec<u8>, &'a T);

    type IntoIter = std::collections::hash_map::Iter<'a, Vec<u8>, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.words.iter()
    }
}

impl<T: TokenType> WordMapTokenVocab<T> {
    /// Shrinks the capacity of the underlying data structures to fit its current size.
    pub fn shrink_to_fit(&mut self) {
        self.words.shrink_to_fit();
    }

    /// The number of words in the vocabulary.
    pub fn len(&self) -> usize {
        self.words.len()
    }

    /// Returns `true` if the vocabulary contains no words.
    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }

    /// Iterate over the words in the vocabulary.
    pub fn iter<'a>(&'a self) -> impl Iterator<Item = (&'a Vec<u8>, &'a T)> + 'a {
        self.words.iter()
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
        self.words.insert(word, token);
    }

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        match self.words.get(chunk) {
            Some(token) => Some(*token),
            None => {
                if chunk.len() == 1 {
                    Some(T::from_u8(chunk[0]).unwrap())
                } else {
                    None
                }
            }
        }
    }

    /// Build word vocabulary from a [`PairMapTokenVocab<T>`].
    pub fn from_pair_vocab(pair_vocab: &PairMapTokenVocab<T>) -> Self {
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
        pair_vocab: &PairMapTokenVocab<T>,
        overwrite: bool,
    ) {
        let skip: Option<AHashSet<T>> = if overwrite {
            None
        } else {
            Some(self.compound_tokens_iter().collect())
        };

        let decoder = PairExpansionDecoder::from_pair_map(&pair_vocab.pairs);
        for token in pair_vocab.compound_tokens_iter() {
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
    pub fn to_pair_vocab(&self) -> PairMapTokenVocab<T> {
        let mut pair_vocab = PairMapTokenVocab::<T>::default();

        let token_to_words: AHashMap<T, &[u8]> = self
            .words
            .iter()
            .map(|(chunk, &token)| (token, chunk.as_ref()))
            .collect();

        for token in self.compound_tokens_iter() {
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

impl<T: TokenType> From<&PairMapTokenVocab<T>> for WordMapTokenVocab<T> {
    fn from(pair_vocab: &PairMapTokenVocab<T>) -> Self {
        Self::from_pair_vocab(pair_vocab)
    }
}

impl<T: TokenType> From<&WordMapTokenVocab<T>> for PairMapTokenVocab<T> {
    fn from(vocab: &WordMapTokenVocab<T>) -> Self {
        vocab.to_pair_vocab()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for WordMapTokenVocab<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.words.values().copied()
    }

    fn max_token(&self) -> T {
        self.words
            .values()
            .max()
            .copied()
            .unwrap_or(T::from_u8(u8::MAX).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_traits::FromPrimitive;
    use std::collections::HashSet;

    #[test]
    fn test_tokens_iter() {
        type T = u32;
        let byte_tokens: Vec<T> = (0..256).map(|b| T::from_usize(b).unwrap()).collect();

        let mut vocab = WordMapTokenVocab::<T>::default();

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(vocab.compound_tokens_iter().collect::<Vec<T>>(), vec![]);

        assert_eq!(&vocab.all_tokens_iter().collect::<Vec<T>>(), &byte_tokens);

        vocab.add_str_word("apple", 300);
        vocab.add_str_word("banana", 301);
        vocab.add_str_word("pear", 302);

        assert_eq!(vocab.max_token(), 302);

        let non_byte_tokens: HashSet<T> = [300, 301, 302].iter().copied().collect();

        let mut combined: HashSet<T> = byte_tokens.iter().copied().collect();
        combined.extend(&non_byte_tokens);

        assert_eq!(
            &vocab.compound_tokens_iter().collect::<HashSet<T>>(),
            &non_byte_tokens,
        );

        assert_eq!(&vocab.all_tokens_iter().collect::<HashSet<T>>(), &combined);
    }

    #[test]
    fn test_lookup_token() {
        type T = u32;
        let mut vocab = WordMapTokenVocab::<T>::default();
        vocab.add_str_word("apple", 300);
        vocab.add_str_word("a", 301);

        assert_eq!(vocab.lookup_token(b"apple"), Some(300));
        assert_eq!(vocab.lookup_token(b"a"), Some(301));
        assert_eq!(vocab.lookup_token(b"b"), Some('b' as u32));
    }

    #[test]
    fn test_build_pair_vocab() {
        type T = u32;
        let mut vocab = WordMapTokenVocab::<T>::default();
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
