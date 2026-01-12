//! # Unified Vocabulary Data

use crate::types::TokenType;
use crate::util::regex::regex_wrapper::RegexWrapperPattern;
use crate::vocab::data::{BPEMapTokenVocab, TokenVocab, WordMapTokenVocab};
use ahash::AHashSet;

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
    pub bpe_vocab: BPEMapTokenVocab<T>,
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
            bpe_vocab: Default::default(),
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

    /// Materialize the tokens in the `bpe_vocab` into the `word_vocab`.
    ///
    /// Leaves tokens which already exist in the `word_vocab`.
    pub fn expand_words_from_bpe(self) -> Self {
        let mut word_vocab = self.word_vocab;

        let tokens: AHashSet<T> = word_vocab.compound_tokens_iter().collect();

        for (w, t) in WordMapTokenVocab::from_bpe(&self.bpe_vocab).words {
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
    pub fn with_bpe_vocab(
        self,
        bpe_vocab: BPEMapTokenVocab<T>,
    ) -> Self {
        Self { bpe_vocab, ..self }
    }

    /// Replace the word vocabulary.
    pub fn with_word_vocab(
        self,
        word_vocab: WordMapTokenVocab<T>,
    ) -> Self {
        Self { word_vocab, ..self }
    }
}

impl<T: TokenType> TokenVocab<T> for UnifiedTokenVocab<T> {
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        let mut tokens = self.bpe_vocab.compound_tokens_iter().collect::<Vec<_>>();

        tokens.extend(self.word_vocab.compound_tokens_iter());

        tokens.into_iter()
    }

    fn max_token(&self) -> T {
        self.bpe_vocab.max_token().max(self.word_vocab.max_token())
    }
}
