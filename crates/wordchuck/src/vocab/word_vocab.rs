//! # Word Map Vocabulary Data

use crate::decoders::pair_decoder::PairExpansionDecoder;
use crate::decoders::token_decoder::TokenDecoder;
use crate::types::{TokenType, WordToTokenMap};
use crate::vocab::pair_vocab::PairMapTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::AHashMap;
use anyhow::Context;
use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use serde::{Deserialize, Serialize};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(bound(serialize = "T: TokenType", deserialize = "T: TokenType"))]
pub struct WordMapTokenVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ &[u8] -> T }``.
    ///
    /// Words in this map take priority over the `binary_pair_map`.
    pub words: WordToTokenMap<T>,
}

impl<T: TokenType> WordMapTokenVocab<T> {
    /// Load a tiktoken vocab file into a [`WordToTokenMap`].
    pub fn load_from_tiktoken_path<P: AsRef<Path>>(path: P) -> anyhow::Result<Self> {
        let mut vocab = WordMapTokenVocab::default();

        let file = std::fs::File::open(path)?;
        let reader = BufReader::new(file);
        for line in reader.lines() {
            let line: String = line?;
            let s = line.as_str();

            let parts = s.splitn(2, ' ').collect::<Vec<&str>>();
            assert_eq!(parts.len(), 2);

            let chunk = parts[0];
            let chunk = BASE64_STANDARD.decode(chunk)?;

            let token: u64 = parts[1].parse()?;
            let token = T::from_u64(token).context("token out of range")?;

            vocab.add_bytes_word(&chunk, token);
        }
        Ok(vocab)
    }

    /// Save this vocab to a tiktoken vocab file.
    pub fn save_to_tiktoken_path<P: AsRef<Path>>(
        &self,
        path: P,
    ) -> anyhow::Result<()> {
        let file = std::fs::File::create(path)?;
        let mut writer = BufWriter::new(file);

        let mut items: Vec<(T, &Vec<u8>)> = self
            .words
            .iter()
            .map(|(chunk, &token)| (token, chunk))
            .collect();
        items.sort_by_key(|(t, _)| *t);

        for (token, chunk) in items {
            writeln!(
                writer,
                "{} {}",
                BASE64_STANDARD.encode(chunk),
                token.to_u64().unwrap()
            )?;
        }

        Ok(())
    }

    /// Add a word to the vocab.
    pub fn add_str_word(
        &mut self,
        word: &str,
        token: T,
    ) {
        self.add_bytes_word(word.as_bytes(), token);
    }

    /// Add a word to the vocab.
    pub fn add_bytes_word(
        &mut self,
        word: &[u8],
        token: T,
    ) {
        self.words.insert(word.to_vec(), token);
    }

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

    /// Build a binary pair map from the word vocabulary.
    pub fn build_pair_vocab(&self) -> PairMapTokenVocab<T> {
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
    fn test_save_load_tiktoken() {
        type T = u32;

        let mut vocab = WordMapTokenVocab::<T>::default();
        vocab.add_str_word("apple", 300);
        vocab.add_str_word("banana", 301);
        vocab.add_str_word("pear", 302);

        tempdir::TempDir::new("vocab_test")
            .and_then(|dir| {
                let path = dir.path().join("vocab.tiktoken");

                vocab
                    .save_to_tiktoken_path(&path)
                    .expect("Failed to save vocab");

                let loaded_vocab = WordMapTokenVocab::<T>::load_from_tiktoken_path(&path)
                    .expect("Failed to load vocab");

                assert_eq!(&vocab, &loaded_vocab);

                Ok(())
            })
            .unwrap();
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

        let pair_vocab = vocab.build_pair_vocab();
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
