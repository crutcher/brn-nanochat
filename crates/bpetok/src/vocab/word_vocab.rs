//! # Word Map Vocabulary Data

use crate::decoder::pair_decoder::PairExpansionDecoder;
use crate::decoder::token_decoder::TokenDecoder;
use crate::types::{TokenType, WordToTokenMap};
use crate::vocab::pair_vocab::PairMapTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use serde::{Deserialize, Serialize};

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Default, Debug, Clone, Serialize, Deserialize)]
#[serde(bound(serialize = "T: TokenType", deserialize = "T: TokenType"))]
pub struct WordMapTokenVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ &[u8] -> T }``.
    ///
    /// Words in this map take priority over the `binary_pair_map`.
    pub words: WordToTokenMap<T>,
}

impl<T: TokenType> WordMapTokenVocab<T> {
    /// Build word vocabulary from a BPE map vocabulary.
    pub fn from_pair_vocab(pair_vocab: &PairMapTokenVocab<T>) -> Self {
        let decoder = PairExpansionDecoder::from_pair_map(&pair_vocab.pairs);
        let mut words = WordToTokenMap::default();
        for token in pair_vocab.compound_tokens_iter() {
            let tokens = [token];
            let chunk = decoder.try_decode_to_bytes(tokens).unwrap();
            words.insert(chunk, token);
        }
        Self { words }
    }

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        self.words.get(chunk).copied()
    }

    /// Save the token vocabulary to a tiktoken vocab file.
    pub fn save_to_tiktoken_vocab(
        &self,
        path: &str,
    ) -> anyhow::Result<()> {
        crate::vocab::tiktoken_io::save_tiktoken_vocab(&self.words, path)
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

        let mut vocab = WordMapTokenVocab::<T> {
            words: WordToTokenMap::default(),
        };

        assert_eq!(vocab.max_token(), 255);

        assert_eq!(vocab.compound_tokens_iter().collect::<Vec<T>>(), vec![]);

        assert_eq!(&vocab.all_tokens_iter().collect::<Vec<T>>(), &byte_tokens);

        vocab.words.insert("apple".as_bytes().to_vec(), 300);
        vocab.words.insert("banana".as_bytes().to_vec(), 301);
        vocab.words.insert("pear".as_bytes().to_vec(), 302);

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
}
