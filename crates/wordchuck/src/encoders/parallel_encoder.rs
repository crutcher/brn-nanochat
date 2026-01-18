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
    /// Inner encoder.
    pub inner: D,

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

#[cfg(test)]
mod tests {
    use crate::decoders::TokenDecoder;
    use crate::encoders::{ParallelEncoder, TokenEncoder, UnifiedVocabEncoder};
    use crate::training::{BinaryPairVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::{TokenVocabIndex, UnifiedTokenVocab};
    use alloc::sync::Arc;
    use compact_str::CompactString;

    #[test]
    fn test_encoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BinaryPairVocabTrainer::new_with_vocab_size(1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let TrainResults {
            word_pattern,
            pair_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let mut vocab: UnifiedTokenVocab<T> =
            UnifiedTokenVocab::new(word_pattern.into()).with_pair_vocab(pair_vocab);

        vocab.specials_vocab_mut().add_str_word("<|HI|>", 3000);

        let special_sample = "hello <|HI|> world";

        let encoder = UnifiedVocabEncoder::<T>::new(Arc::new(vocab));
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let encoder = ParallelEncoder::new(encoder);
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 292);

        let decoder = encoder.inner.to_decoder();
        check_is_send(&decoder);
        check_is_sync(&decoder);

        // Special handling.
        let tokens = encoder.encode(special_sample);
        assert_eq!(
            decoder.try_decode_to_string(tokens).unwrap(),
            special_sample
        );

        for sample in samples {
            let tokens = encoder.encode(sample);
            assert_eq!(decoder.try_decode_to_string(tokens).unwrap(), sample);
        }
    }
}
