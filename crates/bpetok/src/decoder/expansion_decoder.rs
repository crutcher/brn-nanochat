//! # Expansion Decoder

use crate::decoder::TokenDecoder;
use crate::types::{BinaryPairMap, ExpansionMap, TokenType};
use crate::vocab::data::BPEMapTokenVocab;
use ahash::AHashMap;
use std::collections::hash_map;
use std::ops::Range;

/// An [`ExpansionMap`] [`TokenDecoder<T>`].
#[derive(Clone)]
pub struct ExpansionDecoder<T: TokenType> {
    /// Token to pair mapping.
    ///
    /// Does not include byte-tokens.
    pub expansion_map: ExpansionMap<T>,
}

impl<T: TokenType> ExpansionDecoder<T> {
    /// Creates a new Decoder.
    pub fn new(mut expansion_map: ExpansionMap<T>) -> Self {
        expansion_map.shrink_to_fit();
        Self { expansion_map }
    }

    /// Build a [`ExpansionDecoder`] from this [`BPEMapTokenVocab`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(data)))]
    pub fn from_bpe(data: &BPEMapTokenVocab<T>) -> Self {
        Self::from_pair_map(&data.pairs)
    }

    /// Build a [`ExpansionDecoder`] from this [`Tokenizer`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(merge_map)))]
    pub fn from_pair_map(merge_map: &BinaryPairMap<T>) -> Self {
        let expansion_map =
            AHashMap::from_iter(merge_map.iter().map(|(&pair, &token)| (token, pair)));
        Self::new(expansion_map)
    }
}

impl<T: TokenType> TokenDecoder<T> for ExpansionDecoder<T> {
    fn pair_tokens(&self) -> impl Iterator<Item = T> {
        self.expansion_map.keys().copied()
    }

    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, buf, tokens)))]
    fn decode_append(
        &self,
        buf: &mut Vec<u8>,
        tokens: &[T],
    ) {
        let mut tokens = tokens.iter();
        let mut stack: Vec<T> = Vec::with_capacity(16);
        while let Some(t) = stack.pop().or_else(|| tokens.next().copied()) {
            if let Some(b) = t.to_u8() {
                buf.push(b);
                continue;
            }
            let (a, b) = self
                .expansion_map
                .get(&t)
                .expect("Token not found in slice map");
            stack.push(*b);
            stack.push(*a);
        }
    }

    fn size_estimate(&self) -> usize {
        size_of::<hash_map::Entry<T, Range<usize>>>() * self.expansion_map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::TokenEncoder;
    use crate::tokenizer::unified_encoder::ScanningEncoder;
    use crate::types::{check_is_send, check_is_sync};
    use crate::vocab::data::unified::UnifiedTokenVocab;
    use crate::vocab::training::trainer::{BPETokenVocabTrainer, TrainResults};
    use compact_str::CompactString;
    use std::sync::Arc;

    #[test]
    fn test_expansion_decoder() {
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
            bpe_vocab,
        } = options
            .train_vocab_from_sample_iter::<T, K, C, _>(samples.iter())
            .unwrap();

        let vocab: Arc<UnifiedTokenVocab<T>> = UnifiedTokenVocab::new(word_pattern.into())
            .with_bpe_vocab(bpe_vocab)
            .expand_words_from_bpe()
            .into();

        let encoder = ScanningEncoder::<T>::new(vocab.clone(), Default::default());

        let decoder = ExpansionDecoder::from_bpe(&vocab.bpe_vocab);
        check_is_send(&decoder);
        check_is_sync(&decoder);

        for sample in samples {
            let tokens = encoder.encode(sample);
            let decoded = decoder.decode_to_string(&tokens);
            assert_eq!(decoded, sample);
        }
    }
}
