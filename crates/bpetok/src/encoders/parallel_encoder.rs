//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::encoders::text_segmentor::WordRef;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
use crate::vocab::word_vocab::WordMapTokenVocab;

/// Batch-Level Parallel Encoder Wrapper.
///
/// Enables ``rayon`` encoding of batches when available.
#[derive(Clone)]
pub struct ParallelEncoder<T: TokenType, D: TokenEncoder<T>> {
    inner: D,
    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    /// Create a new parallel encoder.
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenVocabIndex<T> for ParallelEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.inner.compound_tokens_iter()
    }
}

impl<T, D> TokenEncoder<T> for ParallelEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn pattern(&self) -> String {
        self.inner.pattern()
    }

    fn special_vocab(&self) -> Option<&WordMapTokenVocab<T>> {
        self.inner.special_vocab()
    }

    fn split_words<'a>(
        &self,
        text: &'a str,
    ) -> Vec<WordRef<'a>> {
        self.inner.split_words(text)
    }

    fn encode_append_word(
        &self,
        word: &str,
        tokens: &mut Vec<T>,
    ) {
        self.inner.encode_append_word(word, tokens)
    }

    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            batch.par_iter().map(|text| self.encode(text)).collect()
        }

        #[cfg(not(feature = "rayon"))]
        {
            batch.iter().map(|text| self.encode(text)).collect()
        }
    }
}
