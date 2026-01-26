//! # Parallel Decoder

use crate::decoders::{TokenDecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// Batch-Level Parallel Decoder Wrapper.
///
/// Enables ``rayon`` decoding of batches when available.
#[derive(Clone)]
pub struct ParallelRayonDecoder<T: TokenType, D: TokenDecoder<T>> {
    /// Wrapped decoder.
    pub inner: D,

    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelRayonDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    /// Create a new parallel token decoders.
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenVocabIndex<T> for ParallelRayonDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    fn unordered_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.inner.unordered_tokens_iter()
    }
}

impl<T, D> TokenDecoder<T> for ParallelRayonDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    fn incremental_decode(
        &self,
        ctx: &mut TokenDecodeContext<T>,
    ) -> bool {
        self.inner.incremental_decode(ctx)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        use rayon::prelude::*;

        batch
            .into_par_iter()
            .map(|tokens| self.try_decode_to_bytes(tokens))
            .collect()
    }

    fn try_decode_batch_to_strings(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<String>> {
        use rayon::prelude::*;

        batch
            .into_par_iter()
            .map(|tokens| {
                let bs = self.try_decode_to_bytes(tokens)?;
                Ok(String::from_utf8_lossy(&bs).to_string())
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoders::DictionaryDecoder;
    use crate::encoders::UnifiedVocabEncoder;
    use crate::encoders::token_encoder::TokenEncoder;
    use crate::training::bpe_trainer::BinaryPairVocabTrainerOptions;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::byte_table::ByteTokenTable;
    use crate::vocab::public::openai::patterns::OA_GPT3_CL100K_WORD_PATTERN;
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use alloc::sync::Arc;
    use compact_str::CompactString;
    use num_traits::FromPrimitive;

    #[test]
    fn test_decoder() {
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

        let vocab: Arc<UnifiedTokenVocab<T>> = trainer
            .train(byte_table.clone())
            .expect("training vocab should succeed")
            .into();

        let encoder = UnifiedVocabEncoder::<T>::init(vocab.clone());

        let decoder = ParallelRayonDecoder::new(DictionaryDecoder::new(vocab.unified_dictionary()));
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples.iter() {
            let tokens = encoder.encode(sample);
            let decoded = decoder.try_decode_to_string(&tokens).unwrap();
            assert_eq!(&decoded, sample);
        }

        let token_batch: Vec<Vec<T>> = samples
            .iter()
            .map(|s| {
                s.as_bytes()
                    .iter()
                    .map(|b| T::from_u8(*b).unwrap())
                    .collect()
            })
            .collect();

        // Test the batch interfaces.
        let string_batch = decoder.try_decode_batch_to_strings(&token_batch).unwrap();
        assert_eq!(string_batch, samples);
    }
}
