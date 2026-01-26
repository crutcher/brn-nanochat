//! # Text Segmentation Configuration
use crate::regex::RegexWrapperPattern;
use crate::types::TokenType;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;

/// Word Split + Special Words Segmentor Configuration
#[derive(Debug, Clone)]
pub struct SegmentationConfig<T: TokenType> {
    /// Regex pattern for word splitting.
    pub pattern: RegexWrapperPattern,

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
    pub fn from_pattern<P>(pattern: P) -> Self
    where
        P: Into<RegexWrapperPattern>,
    {
        Self {
            pattern: pattern.into(),
            specials: SpecialWordsTokenVocab::default(),
        }
    }

    /// Set the split pattern for the text segmentor configuration.
    pub fn with_pattern<P>(
        self,
        pattern: P,
    ) -> Self
    where
        P: Into<RegexWrapperPattern>,
    {
        Self {
            pattern: pattern.into(),
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
        self.pattern.as_str().to_string()
    }

    /// Get the special tokens vocabulary for the text segmentor configuration.
    pub fn special_vocab(&self) -> &SpecialWordsTokenVocab<T> {
        &self.specials
    }
}
