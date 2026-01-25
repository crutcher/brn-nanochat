//! # Unified Token Vocabulary

use crate::regex::RegexWrapperPattern;
use crate::segmentation::segmentation_config::SegmentationConfig;
use crate::types::TokenType;
use crate::vocab::pair_vocab::PairTokenMapVocab;
use crate::vocab::span_vocab::SpanTokenVocab;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;
use crate::vocab::vocab_index::TokenVocabIndex;
use ahash::{AHashMap, AHashSet};

/// Unified token vocabulary.
#[derive(Clone)]
pub struct UnifiedTokenVocab<T: TokenType> {
    /// Text Segmentation Configuration
    pub segmentation: SegmentationConfig<T>,

    /// ``{ Vec<u8> -> T }`` vocabulary.
    pub word_vocab: SpanTokenVocab<T>,

    /// ``{ (T, T) -> T }`` vocabulary.
    pub pair_vocab: PairTokenMapVocab<T>,
}

impl<T: TokenType> UnifiedTokenVocab<T> {
    /// Create a new default token vocabulary.
    ///
    /// # Arguments
    /// * `word_pattern`: Regex pattern for word splitting.
    pub fn new(word_pattern: RegexWrapperPattern) -> Self {
        let segmentation = SegmentationConfig::from_pattern(word_pattern);

        Self {
            segmentation,
            word_vocab: Default::default(),
            pair_vocab: Default::default(),
        }
    }

    /// Mutable reference to the special tokens vocabulary.
    ///
    /// Will create the vocabulary if it doesn't exist.
    pub fn specials_vocab_mut(&mut self) -> &mut SpecialWordsTokenVocab<T> {
        &mut self.segmentation.specials
    }

    /// Replace the word-split regex pattern.
    pub fn with_word_pattern(
        self,
        word_pattern: RegexWrapperPattern,
    ) -> Self {
        Self {
            segmentation: self.segmentation.with_word_pattern(word_pattern),
            ..self
        }
    }

    /// Replace special tokens vocabulary.
    pub fn with_specials(
        self,
        specials: Option<SpecialWordsTokenVocab<T>>,
    ) -> Self {
        Self {
            segmentation: self.segmentation.with_specials(specials),
            ..self
        }
    }

    /// Replace the binary-pair vocabulary.
    pub fn with_pair_vocab(
        self,
        pair_vocab: PairTokenMapVocab<T>,
    ) -> Self {
        Self { pair_vocab, ..self }
    }

    /// Replace the word vocabulary.
    pub fn with_word_vocab(
        self,
        word_vocab: SpanTokenVocab<T>,
    ) -> Self {
        Self { word_vocab, ..self }
    }

    /// Compiled expansion dictionary.
    pub fn compiled_dictionary(&self) -> AHashMap<T, Vec<u8>> {
        let mut tmp = AHashMap::default();

        self.word_vocab.iter().for_each(|(chunk, &token)| {
            tmp.insert(chunk.clone(), token);
        });

        for (span, token) in self.pair_vocab.to_span_pairs() {
            if tmp.contains_key(&span) {
                continue;
            }
            tmp.insert(span, token);
        }

        if let Some(specials) = &self.segmentation.special_vocab() {
            for (chunk, &t) in specials.span_map().iter() {
                tmp.insert(chunk.clone(), t);
            }
        }

        tmp.into_iter()
            .map(|(chunk, token)| (token, chunk))
            .collect()
    }
}

impl<T: TokenType> TokenVocabIndex<T> for UnifiedTokenVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        let mut tokens: AHashSet<T> = self.pair_vocab.unordered_tokens_iter().collect();
        tokens.extend(self.word_vocab.unordered_tokens_iter());
        tokens.into_iter()
    }
}
