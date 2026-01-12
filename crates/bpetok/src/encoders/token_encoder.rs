//! # Token Encoder Trait

use crate::encoders::text_segmentor::WordRef;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType>: TokenVocabIndex<T> + Send + Sync {
    /// Split text using the attached pattern and specials.
    fn split_text<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>>;

    /// Encode a word, and append the resulting tokens to the given token buffer.
    fn encode_append_word(
        &self,
        word: &str,
        tokens: &mut Vec<T>,
    );

    /// Encode bytes into tokens.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    );

    /// Encode text into tokens.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let mut tokens = Vec::with_capacity(text.len());
        self.encode_append(text, &mut tokens);
        tokens
    }

    /// Encode a batch of text into tokens.
    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        batch.iter().map(|s| self.encode(s)).collect()
    }
}
