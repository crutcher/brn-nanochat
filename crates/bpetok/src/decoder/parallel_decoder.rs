//! # Parallel Decoder

use crate::decoder::{DecodeContext, TokenDecoder};
use crate::types::TokenType;
use crate::vocab::TokenVocabIndex;

/// Batch-Level Parallel Decoder Wrapper.
///
/// Enables ``rayon`` decoding of batches when available.
#[derive(Clone)]
pub struct ParallelDecoder<T: TokenType, D: TokenDecoder<T>> {
    inner: D,
    _marker: std::marker::PhantomData<T>,
}

impl<T, D> ParallelDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    /// Create a new parallel token decoder.
    pub fn new(inner: D) -> Self {
        Self {
            inner,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T, D> TokenVocabIndex<T> for ParallelDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    fn compound_tokens_iter(&self) -> impl Iterator<Item = T> {
        self.inner.compound_tokens_iter()
    }
}

impl<T, D> TokenDecoder<T> for ParallelDecoder<T, D>
where
    T: TokenType,
    D: TokenDecoder<T>,
{
    fn incremental_decode(
        &self,
        ctx: &mut DecodeContext<T>,
    ) -> bool {
        self.inner.incremental_decode(ctx)
    }

    fn try_decode_batch_to_bytes(
        &self,
        batch: &[Vec<T>],
    ) -> anyhow::Result<Vec<Vec<u8>>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;

            batch
                .into_par_iter()
                .map(|tokens| self.try_decode_to_bytes(tokens))
                .collect()
        }

        #[cfg(not(feature = "rayon"))]
        batch.iter().map(|t| self.try_decode_to_bytes(t)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decoder::DictionaryDecoder;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::unified_encoder::ScanningEncoder;
    use crate::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::unified_vocab::UnifiedTokenVocab;
    use compact_str::CompactString;
    use num_traits::FromPrimitive;
    use std::sync::Arc;

    #[test]
    fn test_decoder() {
        type T = u16;
        type C = u32;
        type K = CompactString;

        let options = BPETokenVocabTrainer::new_with_vocab_size(1000);

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

        let vocab: Arc<UnifiedTokenVocab<T>> = UnifiedTokenVocab::new(word_pattern.into())
            .with_pair_vocab(pair_vocab)
            .into();

        let encoder = ScanningEncoder::<T>::new(vocab.clone(), Default::default());

        let decoder = ParallelDecoder::new(DictionaryDecoder::new(vocab.compiled_dictionary()));
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
