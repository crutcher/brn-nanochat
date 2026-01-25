//! # Parallel Encoder

use crate::encoders::TokenEncoder;
use crate::segmentation::text_segmentor::WordRef;
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

    fn special_vocab(&self) -> Option<&SpecialWordsTokenVocab<T>> {
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
        use rayon::prelude::*;
        batch.par_iter().map(|text| self.encode(text)).collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::decoders::TokenDecoder;
    use crate::encoders::{TokenEncoder, UnifiedVocabEncoder};
    use crate::rayon::rayon_encoder::ParallelRayonEncoder;
    use crate::training::BinaryPairVocabTrainerOptions;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::{TokenVocabIndex, UnifiedTokenVocab};
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_encoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BinaryPairVocabTrainerOptions::new(OA_GPT3_CL100K_WORD_PATTERN, 1000);

        let samples = vec![
            "hello world",
            "hello san francisco",
            "it's not the heat, it's the salt",
        ];

        let mut trainer = options.init::<K, C>();

        trainer.update_from_samples(samples.iter());

        let byte_table: Arc<ByteTokenTable<T>> = Arc::new(Default::default());

        let mut vocab: UnifiedTokenVocab<T> = trainer
            .train(byte_table.clone())
            .expect("training vocab should succeed");

        vocab.specials_vocab_mut().add_str_word("<|HI|>", 3000);

        let special_sample = "hello <|HI|> world";

        let encoder = UnifiedVocabEncoder::<T>::new(vocab.into());
        check_is_send(&encoder);
        check_is_sync(&encoder);

        let encoder = ParallelRayonEncoder::new(encoder);
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
