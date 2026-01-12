//! # Unified Vocabulary Data

use crate::decoder::dictionary_decoder::DictionaryDecoder;
use crate::types::TokenType;
use crate::util::regex::regex_wrapper::RegexWrapperPattern;
use crate::vocab::data::{PairMapTokenVocab, TokenVocab, WordMapTokenVocab};
use ahash::{AHashMap, AHashSet};

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Regex pattern for word splitting.
    pub word_pattern: RegexWrapperPattern,

    /// Special tokens vocabulary.
    pub specials: Option<WordMapTokenVocab<T>>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub word_vocab: WordMapTokenVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub pair_vocab: PairMapTokenVocab<T>,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Create a new default token vocabulary.
    ///
    /// # Arguments
    /// * `word_pattern`: Regex pattern for word splitting.
    pub fn new(word_pattern: RegexWrapperPattern) -> Self {
        Self {
            word_pattern,
            specials: None,
            word_vocab: Default::default(),
            pair_vocab: Default::default(),
        }
    }

    /// Replace the word-split regex pattern.
    pub fn with_word_pattern(
        self,
        word_pattern: RegexWrapperPattern,
    ) -> Self {
        Self {
            word_pattern,
            ..self
        }
    }

    /// Materialize the tokens in the `pair_vocab` into the `word_vocab`.
    ///
    /// Leaves tokens which already exist in the `word_vocab`.
    pub fn expand_words_from_bpe(self) -> Self {
        let mut word_vocab = self.word_vocab;

        let tokens: AHashSet<T> = word_vocab.compound_tokens_iter().collect();

        for (w, t) in WordMapTokenVocab::from_pair_vocab(&self.pair_vocab).words {
            if !tokens.contains(&t) {
                word_vocab.words.insert(w, t);
            }
        }

        Self { word_vocab, ..self }
    }

    /// Replace special tokens vocabulary.
    pub fn with_specials(
        self,
        specials: Option<WordMapTokenVocab<T>>,
    ) -> Self {
        Self { specials, ..self }
    }

    /// Replace the binary-pair vocabulary.
    pub fn with_pair_vocab(
        self,
        pair_vocab: PairMapTokenVocab<T>,
    ) -> Self {
        Self { pair_vocab, ..self }
    }

    /// Replace the word vocabulary.
    pub fn with_word_vocab(
        self,
        word_vocab: WordMapTokenVocab<T>,
    ) -> Self {
        Self { word_vocab, ..self }
    }

    /// Compiled expansion dictionary.
    pub fn compiled_dictionary(&self) -> AHashMap<T, Vec<u8>> {
        let mut dictionary: AHashMap<T, Vec<u8>> = self
            .clone()
            .expand_words_from_bpe()
            .word_vocab
            .words
            .into_iter()
            .map(|(chunk, token)| (token, chunk))
            .collect();

        if let Some(specials) = &self.specials {
            for (chunk, &t) in &specials.words {
                dictionary.insert(t, chunk.clone());
            }
        }

        dictionary
    }

    /// Compile the unified vocabulary into a dictionary decoder.
    pub fn to_decoder(&self) -> DictionaryDecoder<T> {
        DictionaryDecoder::new(self.compiled_dictionary())
    }
}

impl<T: TokenType> TokenVocab<T> for UnifiedTokenVocab<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        let mut tokens = self.pair_vocab.compound_tokens_iter().collect::<Vec<_>>();

        tokens.extend(self.word_vocab.compound_tokens_iter());

        tokens.into_iter()
    }

    fn max_token(&self) -> T {
        self.pair_vocab.max_token().max(self.word_vocab.max_token())
    }
}
