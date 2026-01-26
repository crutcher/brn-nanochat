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

impl<T: TokenType> From<RegexWrapperPattern> for SegmentationConfig<T> {
    fn from(value: RegexWrapperPattern) -> Self {
        SegmentationConfig::<T>::from_pattern(value)
    }
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
    pub fn with_specials<S>(
        self,
        specials: S,
    ) -> Self
    where
        S: Into<SpecialWordsTokenVocab<T>>,
    {
        let specials = specials.into();
        Self { specials, ..self }
    }

    /// Add a word to the specials.
    pub fn add_str_word(
        &mut self,
        word: &str,
        token: T,
    ) {
        self.specials.add_str_word(word, token);
    }

    /// Add all of the given special words to the specials.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        Self {
            specials: self.specials.with_special_words(special_words),
            ..self
        }
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
