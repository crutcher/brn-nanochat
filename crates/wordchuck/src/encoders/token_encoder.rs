//! # Token Encoder Trait

use crate::segmentation::text_segmentor::SpanRef;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
use crate::vocab::public::size_hints::EXPECTED_BYTES_PER_TOKEN;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;

/// A trait for token encoders.
pub trait TokenEncoder<T: TokenType>: TokenVocabIndex<T> + Send + Sync {
    /// Return the regex pattern used for text splitting.
    fn pattern(&self) -> String;

    /// Return the special vocabulary, if any.
    fn special_vocab(&self) -> &SpecialWordsTokenVocab<T>;

    /// Split text using the attached pattern and specials.
    fn split_spans<'a>(
        &self,
        text: &'a str,
    ) -> Vec<SpanRef<'a>>;

    /// Encode a span appending to a target buffer.
    fn encode_append_span(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    );

    /// Encode bytes into tokens.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, text)))]
    fn encode_append(
        &self,
        text: &str,
        tokens: &mut Vec<T>,
    ) {
        self.split_spans(text).into_iter().for_each(|wr| match wr {
            SpanRef::Normal(w) => self.encode_append_span(w.as_bytes(), tokens),
            SpanRef::Special(s) => {
                tokens.push(self.special_vocab().lookup_token(s.as_bytes()).unwrap());
            }
        });
    }

    /// Encode text into tokens.
    fn encode<S: AsRef<str>>(
        &self,
        text: S,
    ) -> Vec<T> {
        let text = text.as_ref();
        let capacity = text.len() as f64 / (EXPECTED_BYTES_PER_TOKEN * 0.5);
        let mut tokens = Vec::with_capacity(capacity as usize);

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
