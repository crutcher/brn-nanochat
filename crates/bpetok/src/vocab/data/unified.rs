//! # Unified Vocabulary Data

use crate::types::TokenType;
use crate::util::regex::regex_wrapper::RegexPatternLabel;
use crate::vocab::data::{BPEMapTokenVocab, TokenVocab, WordMapTokenVocab};

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Regex pattern for word splitting.
    pub word_pattern: RegexPatternLabel,

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
    pub fn new(word_pattern: RegexPatternLabel) -> Self {
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
        word_pattern: RegexPatternLabel,
    ) -> Self {
        Self {
            word_pattern,
            ..self
        }
    }

    /// Derive words vocabulary from the BPE vocabulary.
    pub fn derive_words(self) -> Self {
        let word_vocab = WordMapTokenVocab::from_bpe(&self.bpe_vocab);
        self.with_word_vocab(word_vocab)
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

    /// Maximum token ID in the vocabulary.
    pub fn max_token(&self) -> T {
        self.bpe_vocab.max_token().max(self.word_vocab.max_token())
    }
}
