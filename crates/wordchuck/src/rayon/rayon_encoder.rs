//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::segmentation::text_segmentor::SpanRef;
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;
use crate::vocab::special_vocab::SpecialWordsTokenVocab;

/// Batch-Level Parallel Encoder Wrapper.
///
/// Enables ``rayon`` encoding of batches when available.
#[derive(Clone)]
pub struct ParallelRayonEncoder<T: TokenType, D: TokenEncoder<T>> {
    /// Inner encoder.
    pub inner: D,

    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelRayonEncoder<T, D>
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

impl<T, D> TokenVocabIndex<T> for ParallelRayonEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.inner.unordered_tokens_iter()
    }
}

impl<T, D> TokenEncoder<T> for ParallelRayonEncoder<T, D>
where
    T: TokenType,
    D: TokenEncoder<T>,
{
    fn pattern(&self) -> String {
        self.inner.pattern()
    }

    fn special_vocab(&self) -> &SpecialWordsTokenVocab<T> {
        self.inner.special_vocab()
    }

    fn split_spans<'a>(
        &self,
        text: &'a str,
    ) -> Vec<SpanRef<'a>> {
        self.inner.split_spans(text)
    }

    fn encode_append_span(
        &self,
        span: &[u8],
        tokens: &mut Vec<T>,
    ) {
        self.inner.encode_append_span(span, tokens)
    }

    fn encode_batch(
        &self,
        batch: &[String],
    ) -> Vec<Vec<T>> {
        use rayon::prelude::*;
        batch.par_iter().map(|text| self.encode(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::{DictionaryDecoder, TokenDecoder};
    use crate::encoders::{MergeHeapVocabEncoder, TokenEncoder};
    use crate::rayon::rayon_encoder::ParallelRayonEncoder;
    use crate::regex::default_regex_supplier;
    use crate::segmentation::SegmentationConfig;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::tooling::testing::new_test_vocab;
    use crate::vocab::{TokenVocabIndex, UnifiedTokenVocab};
    use std::sync::Arc;

    #[test]
    fn test_encoder() {
        type T = u16;

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());
        let segmentation = SegmentationConfig::from_pattern(OA_GPT3_CL100K_WORD_PATTERN);
        let vocab: Arc<UnifiedTokenVocab<T>> = new_test_vocab(byte_table.clone(), segmentation)
            .with_special_words(vec![("<|HI|>", 3000)])
            .into();

        let special_sample = "hello <|HI|> world";

        let encoder = MergeHeapVocabEncoder::<T>::init(vocab.clone(), default_regex_supplier);
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let encoder = ParallelRayonEncoder::new(encoder);
        check_is_send(&encoder);
        check_is_sync(&encoder);

        assert_eq!(encoder.max_token(), 3000);

        let decoder = DictionaryDecoder::from_unified_vocab(vocab);
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
