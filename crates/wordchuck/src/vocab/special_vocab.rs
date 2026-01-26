//! # Special Words Vocabulary

use crate::types::{SpanTokenMap, TokenType};
use crate::vocab::TokenVocabIndex;

/// Token vocabulary as a dictionary map of ``{ Vec<u8> -> T }``.
///
/// This contains no byte:token mappings, or pair mergers.
#[derive(Default, Debug, Clone, PartialEq)]
pub struct SpecialWordsTokenVocab<T: TokenType> {
    /// The regex pattern used for text spl
    /// Map of ``{ Vec<u8> -> T }``.
    span_map: SpanTokenMap<T>,
}

impl<T: TokenType> From<SpanTokenMap<T>> for SpecialWordsTokenVocab<T> {
    fn from(span_map: SpanTokenMap<T>) -> Self {
        Self::new(span_map)
    }
}

impl<T: TokenType> SpecialWordsTokenVocab<T> {
    /// Create a new special words vocab.
    pub fn new(span_map: SpanTokenMap<T>) -> Self {
        Self { span_map }
    }

    /// Get the length of the special words vocab.
    pub fn len(&self) -> usize {
        self.span_map.len()
    }

    /// Returns `true` if the special words vocab contains no words.
    pub fn is_empty(&self) -> bool {
        self.span_map.is_empty()
    }

    /// Get the span => token map.
    pub fn span_map(&self) -> &SpanTokenMap<T> {
        &self.span_map
    }

    /// Add a word to the vocab.
    pub fn add_str_word(
        &mut self,
        word: &str,
        token: T,
    ) {
        self.span_map.insert(word.as_bytes().to_vec(), token);
    }

    /// Extend the vocabulary with the given special words.
    pub fn with_special_words<W, S>(
        self,
        special_words: W,
    ) -> Self
    where
        W: IntoIterator<Item = (S, T)>,
        S: AsRef<str>,
    {
        let mut vocab = self;
        for (word, token) in special_words {
            vocab.add_str_word(word.as_ref(), token);
        }
        vocab
    }

    /// Return the associated token for the word, if any.
    pub fn lookup_token(
        &self,
        chunk: &[u8],
    ) -> Option<T> {
        self.span_map.get(chunk).copied()
    }

    /// Generate all ``(Vec<u8>, T)`` pairs in the vocabulary.
    pub fn to_span_pairs(&self) -> impl Iterator<Item = (Vec<u8>, T)> {
        self.span_map
            .iter()
            .map(|(chunk, &token)| (chunk.clone(), token))
    }
}

impl<T: TokenType> TokenVocabIndex<T> for SpecialWordsTokenVocab<T> {
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.span_map.values().copied()
    }
}
