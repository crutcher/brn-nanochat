//! # Text Segmentation Configuration
use crate::regex::RegexWrapperPattern;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;

/// Word Split + Special Words Segmentor Configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig<T: TokenType> {
    /// Regex pattern for word splitting.
    pub word_pattern: RegexWrapperPattern,

    /// Special tokens vocabulary.
    pub specials: SpecialWordsTokenVocab<T>,
}

impl<T: TokenType> SegmentationConfig<T> {
    /// Create a new text segmentor configuration with the given word pattern.
    ///
    /// Will contain an empty list of specials.
    pub fn from_pattern(word_pattern: RegexWrapperPattern) -> Self {
        Self {
            word_pattern,
            specials: SpecialWordsTokenVocab::default(),
        }
    }

    /// Set the word pattern for the text segmentor configuration.
    pub fn with_word_pattern(
        self,
        word_pattern: RegexWrapperPattern,
    ) -> Self {
        Self {
            word_pattern,
            ..self
        }
    }

    /// Replace special tokens vocabulary.
    pub fn with_specials(
        self,
        specials: Option<SpecialWordsTokenVocab<T>>,
    ) -> Self {
        let specials = specials.unwrap_or_default();
        Self { specials, ..self }
    }

    /// Get the word pattern for the text segmentor configuration.
    pub fn pattern(&self) -> String {
        self.word_pattern.as_str().to_string()
    }

    /// Get the special tokens vocabulary for the text segmentor configuration.
    pub fn special_vocab(&self) -> Option<&SpecialWordsTokenVocab<T>> {
        if self.specials.is_empty() {
            None
        } else {
            Some(&self.specials)
        }
    }
}
